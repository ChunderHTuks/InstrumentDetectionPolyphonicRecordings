# `docs/`

| File | Overview |
|---|---|
| `22659511_ChunderH_FINALREPORT.pdf` | The full EPR402 capstone report: literature review, architecture trade-off analysis (Mel-spectrogram/MFCC/HCQT, ResNet vs. VGG), dataset compilation and stratification, from-scratch implementation and verification of every major component, onset-detection design, deployment details, and full qualification-test results. |
| `figures/system_block_diagram.png` | System block diagram, showing off-the-shelf vs. developed components and the data flow from microphone through to detected onsets. Embedded in the top-level README's "How it works" section. |
| `figures/multihead.png` | The multi-head CNN architecture diagram: shared backbone, four family heads, concatenated output. Embedded in the top-level README's "Model" description. |
| `figures/cqt_difference.png` | Verification plot showing the per-bin difference (dB) between the custom C CQT implementation and `librosa.cqt` on the same audio. Embedded in the top-level README's "Verification methodology" section. |
