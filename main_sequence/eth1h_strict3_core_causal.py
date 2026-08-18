from main_sequence import eth15m_conservative_replay as eth
from main_sequence.eth_causal_anchor import build_eth_anchors_causal
from main_sequence import eth1h_strict3_core as runner

eth.build_anchors = build_eth_anchors_causal
runner.eth.build_anchors = build_eth_anchors_causal

if __name__ == "__main__":
    runner.main()
