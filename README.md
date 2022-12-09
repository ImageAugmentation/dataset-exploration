<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">
graph TD;
    SCP(SimpleCopyPaste<br>CVPR,2021)
    click SCP "https://arxiv.org/abs/2012.07177" "Simple copy-paste is a strong data augmentation method for instance segmentation" _blank

    Geosim(Geosim<br>CVPR,2021)
    click Geosim "https://arxiv.org/abs/2101.06543" "GeoSim: Realistic Video Simulation via Geometry-Aware Composition for Self-Driving" _blank

    ScaleAwareAA(ScaleAwareAutoAugment<br>CVPR,2021)
    click ScaleAwareAA "https://arxiv.org/pdf/2103.17220v1.pdf" "Scale-aware Automatic Augmentation for Object Detection" _blank

    PlaceNet(PlaceNet<br>ECCV,2020)
    click PlaceNet "https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf" "PlaceNet: Learning Object Placement by Inpainting for Compositional Data Augmentation" _blank

    SwitchingInstance(SwitchingInstance<br>CoRR?,2019)
    click SwitchingInstance "https://arxiv.org/abs/1906.00358" "Data Augmentation for Object Detection via Progressive and Selective Instance-Switching" _blank

    InstaBoost(InstaBoost<br>ICCV,2019)
    click InstaBoost "https://arxiv.org/abs/1908.07801v1" "InstaBoost: Boosting Instance Segmentation via Probability Map Guided Copy-Pasting" _blank

    Learnin2SegmentCutPaste(LearningToSegment<br>ECCV,2018)
    click Learnin2SegmentCutPaste "https://openaccess.thecvf.com/content_ECCV_2018/html/Tal_Remez_Learning_to_Segment_ECCV_2018_paper.html" "Learning to Segment via Cut-and-Paste" _blank

    ContextualPaste(ContextualCopyPaste<br>ECCV,2018)
    click ContextualPaste "https://arxiv.org/pdf/1809.02492.pdf" "On the Importance of Visual Context for Data Augmentation in Scene Understanding" _blank

    CutPasteLearn(CutPaste&Learn<br>ICCV,2017)
    click CutPasteLearn "https://arxiv.org/abs/1708.01642?context=cs" "Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection" _blank

    ARmeetCV(ARmeetCV<br>-,2017)
    click ARmeetCV "https://arxiv.org/abs/1708.01566" "Augmented Reality Meets Computer Vision : Efficient Data Generation for Urban Driving Scenes" _blank


    InstaBoost-->|cited by|SCP;
    CutPasteLearn-->SCP;
    Learnin2SegmentCutPaste-->SCP;
    ContextualPaste-->SCP;

    ARmeetCV-->Geosim
    CutPasteLearn-->Geosim

    InstaBoost-->ScaleAwareAA
    CutPasteLearn-->ScaleAwareAA
    SwitchingInstance-->ScaleAwareAA
    ContextualPaste-->ScaleAwareAA

    InstaBoost-->PlaceNet
    ContextualPaste-->PlaceNet
    CutPasteLearn-->PlaceNet
    SwitchingInstance-->PlaceNet

    ContextualPaste-->SwitchingInstance

    ContextualPaste-->InstaBoost;
    CutPasteLearn-->InstaBoost;

    ARmeetCV-->Learnin2SegmentCutPaste
     
</div>
