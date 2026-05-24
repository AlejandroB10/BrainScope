# Data

This folder is **not** tracked in git: the DICOM volumes are large and live
outside the repository. Each acquisition is a single multi-frame DICOM file:

```
data/
├── raw/
│   ├── e_1_BRAIN_DINAMIC_COLINA.dcm   # Dynamic PET (multi-frame, ~212 MB)
│   └── AX_3D_T1.dcm                   # MR T1 reference volume (~20 MB)
└── README.md
```

## Source

Both studies are provided through the course material. The dataset can be
downloaded from the link given in the project proposal
(`11763_ProjectProposal.pdf`, Objective 1.a).

## Relevant DICOM headers

The dynamic PET pixel array must be reorganized according to the following
headers before any temporal or spatial analysis:

| Tag         | Field                       | Use                                                  |
|-------------|-----------------------------|------------------------------------------------------|
| (0028,0008) | Number of Frames            | Number of temporal frames in the dynamic acquisition |
| (0028,0010) | Rows                        | Volume rows                                          |
| (0028,0011) | Columns                     | Volume columns                                       |
| (0018,0088) | Spacing Between Slices      | Z spacing in millimetres                             |
| (0028,0030) | Pixel Spacing               | XY spacing in millimetres                            |
| (0055,1002) | Frame Positions Vector      | Z-position of each frame slice                       |
| (0055,1001) | Frame Start Times Vector    | Onset of each temporal frame                         |
| (0055,1004) | Frame Durations             | Duration of each temporal frame in milliseconds      |
