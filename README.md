We have anonymized the information!

# Introduction
PhysiqueFrame conducts physique aesthetics assessment on a physique image and predicts scores across three dimensions of physique aesthetics.

# Environment Installation
see requirement.txt

# How to Check the Code
1. download PhysiqueFrame
2. download datasets from anonymous url https://drive.google.com/file/d/16kM10BDbsX93S7PJ_ofaS0fe7vHUzqtt/view?usp=drive_link
3. download pth from anonymous url https://drive.google.com/file/d/1ubuIdoUT68kGERLAcuyVfxl4L05jLmd8/view?usp=drive_link :
    1) put PhysiqueFrame_pth/else/* into PhysiqueFrame/models_/pam/pretrained/
    2) put PhysiqueFrame_pth/pretrained_encoder_pth/* into PhysiqueFrame/SMPLer_X/pretrained_models/
    3) put PhysiqueFrame_pth/User-ISFJ/* into your directory (PhysiqueFrame_pth also provides User-ESFJ and User-ISTJ), and update the 'path_to_model_weight' parameter in option_user.py.
4. run main_nni_gnn_pam_user.py
