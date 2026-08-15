from __future__ import annotations

# The original full-fetch baseline has a mechanical field-name typo:
# Market defines `label_up`, while core.build_examples reads `m.label`.
# Add the intended read-only alias only so the June equivalence baseline can
# reach feature construction.  No label value or statistical logic changes.
import prejuly_5m_official as core

if not hasattr(core.Market, "label"):
    core.Market.label = property(lambda self: self.label_up)

import prejuly_5m_official_stream as stream

if __name__ == "__main__":
    stream.main()
