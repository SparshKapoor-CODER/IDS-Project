#!/usr/bin/env python3
"""
PCAP to IDS Pipeline - Enhanced Version
========================================
Converts network traffic from .pcap files to IDS predictions using a trained TFLite model.

Usage:
    python app.py --pcap <path_to_pcap> [--output <output_csv>] [--artifacts <dir>]

The pipeline:
1. Parses .pcap with scapy (IPv4/IPv6 packets only)
2. Aggregates packets into bidirectional flows
3. Computes 42 UNSW‑NB15 flow features
4. Preprocesses features (encoding, scaling, feature selection)
5. Runs inference with a quantized TFLite model
6. Saves predictions and prints summary
"""

import os
import json
import pickle
import argparse
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Suppress scapy's IPv6 warning if not needed
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from scapy.all import rdpcap, IP, IPv6, TCP, UDP, ICMP
except ImportError:
    raise ImportError("scapy is required. Install with: pip install scapy")

# TensorFlow Lite runtime
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        logging.error("No TensorFlow or tflite_runtime found. Install with: pip install tflite-runtime")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PcapFeatureExtractor:
    """
    Extract UNSW-NB15 features from a pcap file.
    Packets are grouped into flows based on (src_ip, src_port, dst_ip, dst_port, protocol).
    """

    # Complete list of 42 raw features (order must match training)
    RAW_FEATURES = [
        'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
        'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
        'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
        'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
        'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
    ]

    # Port -> service name mapping
    SERVICE_MAP = {
        80: 'http', 443: 'https', 21: 'ftp', 22: 'ssh',
        53: 'dns', 25: 'smtp', 143: 'imap', 110: 'pop3',
        23: 'telnet', 123: 'ntp', 161: 'snmp', 445: 'microsoft-ds'
    }

    def __init__(self):
        self.flows = defaultdict(self._init_flow)
        self.all_timestamps = []

    def _init_flow(self) -> Dict:
        """Initialize a flow dictionary with default values."""
        return {
            'start_time': None,
            'end_time': None,
            'src_pkts': 0,
            'dst_pkts': 0,
            'src_bytes': 0,
            'dst_bytes': 0,
            'proto': None,
            'state': '-',
            'service': '-',
            'src_ttl': 64,
            'dst_ttl': 64,
            'src_loss': 0,
            'dst_loss': 0,
            'tcp_flags': set(),
            'src_jitter': 0,
            'dst_jitter': 0,
            'src_interpacket': [],
            'dst_interpacket': [],
            'src_window': 0,
            'dst_window': 0,
            'syn_ack_time': None,
            'ack_dat_time': None,
            'src_timestamps': [],
            'dst_timestamps': [],
            'response_body_len': 0,
            'trans_depth': 0,
            'http_methods': 0,
        }

    @staticmethod
    def _get_service_from_port(port: int) -> str:
        """Map port number to service name, default '-'."""
        return PcapFeatureExtractor.SERVICE_MAP.get(port, '-')

    @staticmethod
    def _update_tcp_state(flags: int) -> str:
        """Map TCP flags to connection state (UNSW-NB15 style)."""
        if flags is None:
            return '-'
        if flags & 0x02:      # SYN
            return 'CON'
        elif flags & 0x10:    # ACK
            return 'ACC'
        elif flags & 0x01:    # FIN
            return 'FIN'
        elif flags & 0x04:    # RST
            return 'RST'
        else:
            return 'OTH'

    def extract_features_from_pcap(self, pcap_path: str) -> pd.DataFrame:
        """
        Parse a pcap file and return a DataFrame with one row per flow.

        Args:
            pcap_path: Path to the .pcap file.

        Returns:
            DataFrame with 42 columns (RAW_FEATURES).
        """
        logger.info(f"Reading pcap file: {pcap_path}")
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            raise ValueError(f"Failed to read pcap file: {e}")

        logger.info(f"Total packets in pcap: {len(packets)}")
        ip_packet_count = 0

        for pkt in packets:
            # Extract IP layer (IPv4 or IPv6)
            ip_layer = None
            if IP in pkt:
                ip_layer = pkt[IP]
                proto = 'tcp' if TCP in pkt else 'udp' if UDP in pkt else 'icmp'
            elif IPv6 in pkt:
                ip_layer = pkt[IPv6]
                proto = 'tcp' if TCP in pkt else 'udp' if UDP in pkt else 'icmp'
            else:
                continue   # Not an IP packet

            ip_packet_count += 1
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            ttl = ip_layer.hlim if isinstance(ip_layer, IPv6) else ip_layer.ttl
            timestamp = float(pkt.time)
            payload_len = len(bytes(pkt.payload))

            src_port = None
            dst_port = None
            tcp_flags = None

            if TCP in pkt:
                tcp_layer = pkt[TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                tcp_flags = tcp_layer.flags
                proto = 'tcp'
            elif UDP in pkt:
                udp_layer = pkt[UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
                proto = 'udp'
            elif ICMP in pkt:
                src_port = 0
                dst_port = 0
                proto = 'icmp'
            else:
                continue

            # Flow key: (src_ip, src_port, dst_ip, dst_port, proto)
            flow_key = (src_ip, src_port, dst_ip, dst_port, proto)
            flow = self.flows[flow_key]

            # Initialize timestamps
            if flow['start_time'] is None:
                flow['start_time'] = timestamp
            flow['end_time'] = timestamp
            flow['proto'] = proto
            flow['src_ttl'] = ttl
            flow['service'] = self._get_service_from_port(dst_port) if dst_port else '-'

            # Forward direction (src -> dst)
            flow['src_pkts'] += 1
            flow['src_bytes'] += payload_len
            flow['src_timestamps'].append(timestamp)

            if tcp_flags:
                flow['tcp_flags'].add(tcp_flags)
                flow['state'] = self._update_tcp_state(tcp_flags)

            self.all_timestamps.append(timestamp)

        logger.info(f"IP packets processed: {ip_packet_count}")
        logger.info(f"Number of flows identified: {len(self.flows)}")

        # Build DataFrame from flows
        flow_records = []
        for (src_ip, src_port, dst_ip, dst_port, proto), flow in self.flows.items():
            features = self._compute_flow_features(flow, src_ip, src_port, dst_ip, dst_port, proto)
            flow_records.append(features)

        df = pd.DataFrame(flow_records)

        # Ensure all expected features exist
        for feat in self.RAW_FEATURES:
            if feat not in df.columns:
                df[feat] = 0

        # Reorder columns
        return df[self.RAW_FEATURES]

    def _compute_flow_features(self, flow: Dict,
                               src_ip: str, src_port: int,
                               dst_ip: str, dst_port: int,
                               proto: str) -> Dict:
        """Compute all 42 UNSW-NB15 features for a single flow."""
        duration = (flow['end_time'] - flow['start_time']) if flow['start_time'] else 0.0
        src_pkts = flow['src_pkts']
        dst_pkts = flow['dst_pkts']
        src_bytes = flow['src_bytes']
        dst_bytes = flow['dst_bytes']
        rate = (src_bytes + dst_bytes) / duration if duration > 0 else 0.0
        src_load = (src_bytes * 8) / duration if duration > 0 else 0.0
        dst_load = (dst_bytes * 8) / duration if duration > 0 else 0.0

        # Interpacket arrival times (milliseconds)
        src_inter = np.diff(flow['src_timestamps']) if len(flow['src_timestamps']) > 1 else [0.0]
        dst_inter = flow['dst_interpacket'] if flow['dst_interpacket'] else [0.0]
        sinpkt = np.mean(src_inter) * 1000.0 if len(src_inter) > 0 else 0.0
        dinpkt = np.mean(dst_inter) * 1000.0 if len(dst_inter) > 0 else 0.0

        # Jitter (standard deviation of interpacket times)
        sjit = np.std(src_inter) * 1000.0 if len(src_inter) > 1 else 0.0
        djit = np.std(dst_inter) * 1000.0 if len(dst_inter) > 1 else 0.0

        # Mean packet size
        smean = src_bytes / src_pkts if src_pkts > 0 else 0.0
        dmean = dst_bytes / dst_pkts if dst_pkts > 0 else 0.0

        # Derived counters (simplified for single‑pcap analysis)
        # In a full system these would be computed across all flows in the capture
        ct_srv_src = 1           # flows with same service & source
        ct_state_ttl = 1         # flows with same state & TTL
        ct_dst_ltm = 1           # flows to same destination
        ct_src_dport_ltm = 1     # flows from same source to same dest port
        ct_dst_sport_ltm = 1     # flows to same dest from same source port
        ct_dst_src_ltm = 1       # flows between same src‑dst pair
        ct_src_ltm = 1           # flows from same source
        ct_srv_dst = 1           # flows with same service & destination
        is_sm_ips_ports = 1 if (src_ip == dst_ip and src_port == dst_port) else 0

        # Service‑specific flags
        service = flow['service']
        is_ftp_login = 1 if service == 'ftp' else 0
        ct_ftp_cmd = 1 if service == 'ftp' else 0
        ct_flw_http_mthd = flow['http_methods']

        # TCP time features (placeholders – would require SYN/SYN‑ACK tracking)
        tcp_rtt = 0.0
        synack = 0.0
        ackdat = 0.0

        # Placeholder values for fields not extracted
        stcpb = 0
        dtcpb = 0
        swin = flow['src_window']
        dwin = flow['dst_window']
        trans_depth = flow['trans_depth']
        response_body_len = flow['response_body_len']
        sloss = flow['src_loss']
        dloss = flow['dst_loss']

        return {
            'dur': duration,
            'proto': proto.lower(),
            'service': service,
            'state': flow['state'],
            'spkts': src_pkts,
            'dpkts': dst_pkts,
            'sbytes': src_bytes,
            'dbytes': dst_bytes,
            'rate': rate,
            'sttl': flow['src_ttl'],
            'dttl': flow['dst_ttl'],
            'sload': src_load,
            'dload': dst_load,
            'sloss': sloss,
            'dloss': dloss,
            'sinpkt': sinpkt,
            'dinpkt': dinpkt,
            'sjit': sjit,
            'djit': djit,
            'swin': swin,
            'stcpb': stcpb,
            'dtcpb': dtcpb,
            'dwin': dwin,
            'tcprtt': tcp_rtt,
            'synack': synack,
            'ackdat': ackdat,
            'smean': smean,
            'dmean': dmean,
            'trans_depth': trans_depth,
            'response_body_len': response_body_len,
            'ct_srv_src': ct_srv_src,
            'ct_state_ttl': ct_state_ttl,
            'ct_dst_ltm': ct_dst_ltm,
            'ct_src_dport_ltm': ct_src_dport_ltm,
            'ct_dst_sport_ltm': ct_dst_sport_ltm,
            'ct_dst_src_ltm': ct_dst_src_ltm,
            'is_ftp_login': is_ftp_login,
            'ct_ftp_cmd': ct_ftp_cmd,
            'ct_flw_http_mthd': ct_flw_http_mthd,
            'ct_src_ltm': ct_src_ltm,
            'ct_srv_dst': ct_srv_dst,
            'is_sm_ips_ports': is_sm_ips_ports,
        }


class IDSPredictor:
    """Load trained IDS artifacts and run predictions on extracted features."""

    def __init__(self, artifacts_dir: str = 'artifacts'):
        """
        Args:
            artifacts_dir: Directory containing:
                - models/IDSmodel.tflite
                - scaler.pkl
                - label_encoders.pkl
                - selected_features.json
        """
        self.artifacts_dir = artifacts_dir

        # Load TFLite model
        model_path = os.path.join(artifacts_dir, 'models', 'IDSmodel.tflite')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        logger.info(f"Loaded TFLite model from {model_path}")

        # Load preprocessing artifacts
        with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'rb') as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(artifacts_dir, 'selected_features.json'), 'r') as f:
            self.selected_features = json.load(f)

        logger.info(f"Loaded scaler, label encoders, and {len(self.selected_features)} selected features")

        # Feature names (must match the order used in training)
        self.raw_feature_names = PcapFeatureExtractor.RAW_FEATURES

    def preprocess_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw feature DataFrame and return predictions.

        Args:
            df: DataFrame with 42 raw features (columns match self.raw_feature_names)

        Returns:
            DataFrame with original columns plus 'prediction' and 'confidence'.
        """
        results = []

        for idx, row in df.iterrows():
            try:
                data = row.to_dict()

                # Encode categorical features
                for col in ['proto', 'service', 'state']:
                    val = str(data.get(col, '-')).strip()
                    le = self.encoders.get(col)
                    if le:
                        try:
                            data[col] = le.transform([val])[0]
                        except ValueError:
                            # Unknown label – treat as -1 (or most frequent)
                            data[col] = -1
                    else:
                        data[col] = -1

                # Build numeric row in correct order
                full_row = []
                for feat in self.raw_feature_names:
                    try:
                        full_row.append(float(data.get(feat, 0)))
                    except (ValueError, TypeError):
                        full_row.append(0.0)

                X_raw = np.array(full_row, dtype=np.float32).reshape(1, -1)

                # Scale
                X_scaled = self.scaler.transform(X_raw)

                # Select features
                idx_map = {name: i for i, name in enumerate(self.raw_feature_names)}
                X_reduced = np.array(
                    [X_scaled[0, idx_map[f]] for f in self.selected_features],
                    dtype=np.float32
                ).reshape(1, -1)

                # Reshape if model expects 3D (e.g., LSTM)
                if len(self.input_details['shape']) == 3:
                    X_input = X_reduced.reshape(1, len(self.selected_features), 1)
                else:
                    X_input = X_reduced

                # Quantize if model is int8
                if self.input_details['dtype'] == np.int8:
                    in_scale, in_zero = self.input_details['quantization']
                    X_input = (X_input / in_scale + in_zero).clip(-128, 127).astype(np.int8)

                # Run inference
                self.interpreter.set_tensor(self.input_details['index'], X_input)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details['index'])[0][0]

                # Dequantize and apply sigmoid if needed
                if self.output_details['dtype'] == np.int8:
                    out_scale, out_zero = self.output_details['quantization']
                    prob = (float(out) - out_zero) * out_scale
                else:
                    prob = float(out)

                # Ensure probability in [0,1]
                if not (0 <= prob <= 1):
                    prob = 1.0 / (1.0 + np.exp(-prob))

                prediction = 'Attack' if prob >= 0.5 else 'Normal'
                results.append({'row': idx, 'prediction': prediction, 'confidence': prob})

            except Exception as e:
                logger.error(f"Prediction failed for flow {idx}: {e}")
                results.append({'row': idx, 'prediction': 'Error', 'confidence': 0.0})

        results_df = pd.DataFrame(results)
        output_df = df.copy()
        output_df['prediction'] = results_df['prediction']
        output_df['confidence'] = results_df['confidence']
        return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert pcap file to IDS predictions (Normal / Attack)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py --pcap traffic.pcap
    python app.py --pcap traffic.pcap --output results.csv --artifacts ./my_artifacts

Note: The pcap must contain IP packets (IPv4 or IPv6). Non-IP traffic (e.g., raw Bluetooth)
      will be ignored, and the model will not produce meaningful predictions.
        """
    )
    parser.add_argument('--pcap', required=True, help='Path to input .pcap file')
    parser.add_argument('--output', default='ids_predictions.csv', help='Output CSV file (default: ids_predictions.csv)')
    parser.add_argument('--artifacts', default='artifacts', help='Directory containing model and preprocessors (default: artifacts)')

    args = parser.parse_args()

    if not os.path.exists(args.pcap):
        logger.error(f"PCAP file not found: {args.pcap}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("PCAP to IDS Prediction Pipeline")
    logger.info("=" * 70)

    # Step 1: Extract features
    logger.info("Step 1: Extracting flow features from pcap...")
    extractor = PcapFeatureExtractor()
    features_df = extractor.extract_features_from_pcap(args.pcap)

    if features_df.empty:
        logger.warning("No IP flows found in pcap. Exiting.")
        sys.exit(0)

    logger.info(f"Extracted {len(features_df)} flows")

    # Step 2: Load model and predict
    logger.info("Step 2: Loading model and running predictions...")
    predictor = IDSPredictor(artifacts_dir=args.artifacts)
    results_df = predictor.preprocess_and_predict(features_df)

    # Step 3: Save results
    logger.info(f"Step 3: Saving predictions to {args.output}")
    results_df.to_csv(args.output, index=False)

    # Summary
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    total = len(results_df)
    attacks = (results_df['prediction'] == 'Attack').sum()
    normal = (results_df['prediction'] == 'Normal').sum()
    errors = (results_df['prediction'] == 'Error').sum()
    logger.info(f"Total flows analyzed : {total}")
    logger.info(f"  Normal : {normal} ({100*normal/total:.1f}%)")
    logger.info(f"  Attack : {attacks} ({100*attacks/total:.1f}%)")
    if errors:
        logger.warning(f"  Errors : {errors}")
    if 'confidence' in results_df.columns:
        conf = results_df[results_df['prediction'] != 'Error']['confidence'].mean()
        logger.info(f"Average confidence    : {conf:.4f}")

    logger.info(f"Output saved to: {os.path.abspath(args.output)}")
    logger.info("Pipeline completed successfully.")


if __name__ == '__main__':
    import sys
    main()