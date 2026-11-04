# Citations

These are the works referenced in the project's notes. A link is given only where a source document gave one; no
reference was added that the sources don't contain. The "Cited in" column names the source; all except
`CLAUDE.md` and the code are outside git.

**Source key:**

- **OV** = `review/open-vocab-research/PLAN.md`. It marks each claim [verified] (read in the paper) or [not
  verified].
- **LIT** = the occlusion-noise literature notes, `unbooru-scratch/occlusion-noise/literature.md` on the scratch
  drive. It marks numbers (v) checked or (m) from memory.
- **POOL** = `review/pooling/PLAN.md`.
- **CL** = `CLAUDE.md`.

## Architecture and prior art

| Work | Link | Cited in | Used for |
|---|---|---|---|
| SigLIP (sigmoid loss for image-text pre-training) | not given | CL | The sigmoid contrastive loss and the learnable logit scale/bias. |
| JoyTag | not given | CL | Reference implementation studied. |
| WD14 tagger; DeepDanbooru | not given | CL | Prior art with a fixed classification head, which this project moves past. |
| ConvNeXt (LayerScale, published small-std init) | not given | CL, commits | The image tower; LayerScale. |
| SimSiam | not given | CL | The self-supervised consistency loss. |
| MaskCLIP: Zhou, Loy, Dai, "Extract Free Dense Labels from CLIP", ECCV 2022 | [arXiv:2112.01071](https://arxiv.org/abs/2112.01071) | CL, OV | Heatmaps from the dot product at each location. |
| CAM + DenseCRF refinement (weakly supervised segmentation) | not given | CL | The bilateral-filter box refinement as its label-free analogue. |

## Context bias and erased views

| Work | Link | Cited in | Used for |
|---|---|---|---|
| Singh et al., "Don't Judge an Object by Its Context", CVPR 2020 | [arXiv:2001.03152](https://arxiv.org/abs/2001.03152) | LIT, POOL, CL | Feature split; exclusive vs. co-occur evaluation; the warning that heavy out-of-context weighting costs co-occur mAP. |
| Kim et al. 2021, "[Re] Don't Judge an Object by Its Context" (reproduction) | not given | POOL | Shows the feature split and the weighting each do little alone. |
| Shetty, Schiele, Fritz, "Not Using the Car to See the Sidewalk", CVPR 2019 | [arXiv:1812.06707](https://arxiv.org/abs/1812.06707) | LIT, CL | Ranking hinge on object-removed views: the template for erased-view training. |
| GAIN: Li et al., "Tell Me Where to Look", CVPR 2018 | [arXiv:1802.10171](https://arxiv.org/abs/1802.10171) | LIT, CL | Erased view as a negative (the `negative` erasure variant). |
| CSS: Chen et al., "Counterfactual Samples Synthesizing for Robust VQA", CVPR 2020 | [CVF open access](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.html) | LIT, CL | Counterfactual negatives. |
| Adversarial erasing: Wei et al., CVPR 2017 | [arXiv:1703.08448](https://arxiv.org/abs/1703.08448) | LIT | Erasing with the label kept positive; rewards context, the opposite direction. |
| Hide-and-Seek: Singh & Lee, ICCV 2017 | not given | LIT | Random patch hiding. |
| ACoL; ADL (Choe & Shim, CVPR 2019); SeeNet (Hou et al., NeurIPS 2018) | not given | LIT | Guards against erasing the wrong region. |
| CONTA: Zhang et al., "Causal Intervention for Weakly-Supervised Semantic Segmentation", NeurIPS 2020 | [arXiv:2009.12547](https://arxiv.org/abs/2009.12547) | LIT, OV | Context as a confounder; close in spirit to the rejected context prior. |
| CDA: Su et al., context decoupling augmentation, ICCV 2021 | [arXiv:2103.01795](https://arxiv.org/abs/2103.01795) | OV | Pasting instances out of context; the pasted-twins idea. |
| Copy-paste (Dwibedi et al., ICCV 2017; Ghiasi et al., CVPR 2021, Simple Copy-Paste) | not given | LIT | Mask-based pasting; mosaics as the mask-free version. |
| TDE (Tang et al. 2020); Kaushik et al. 2020 (counterfactually augmented pairs) | not given (marked [not verified] in OV) | OV, POOL | The same-source pairs arm. |
| RRR: Ross et al., "Right for the Right Reasons", IJCAI 2017 | [arXiv:1703.03717](https://arxiv.org/abs/1703.03717) | LIT | Input-gradient penalties; not recommended. |
| CDEP: Rieger et al., ICML 2020 | [arXiv:1909.13584](https://arxiv.org/abs/1909.13584) | LIT | Same family as RRR. |
| GradMask (Simpson et al., 2019) | not given | LIT | Same family. |
| Salient ImageNet: Singla & Feizi, ICLR 2022 | [arXiv:2110.04301](https://arxiv.org/abs/2110.04301) | LIT | Noise instead of a flat blank, as an occlusion fill. |
| Core Risk Minimization | [arXiv:2203.15566](https://arxiv.org/abs/2203.15566) | LIT | Relative core sensitivity. |
| RISE deletion/insertion (Petsiuk et al., BMVC 2018) | not given | LIT | Deletion metrics. |
| JTT: Liu et al., ICML 2021 | [arXiv:2107.09044](https://arxiv.org/abs/2107.09044) | LIT | Upweighting errors. |
| DFR: Kirichenko et al., ICLR 2023; LaBonte et al., NeurIPS 2023 | not given | LIT | The last-layer refit tried in `ctxfp/`. |

## Missing labels and negatives

| Work | Link | Cited in | Used for |
|---|---|---|---|
| ASL (asymmetric loss) | not given | CL, code | The loss for unverified negatives (`--negative-gamma`, `--negative-clip`). |
| Kim et al., "Large Loss Matters in Weakly Supervised Multi-Label Classification", CVPR 2022 | [arXiv:2206.03740](https://arxiv.org/abs/2206.03740) | LIT, OV, POOL | Large-loss rejection (plan #5, not built). |
| Hill / SPLC: Zhang et al., 2021 | [arXiv:2112.07368](https://arxiv.org/abs/2112.07368) | LIT, POOL | `--negative-loss hill` (built, untested). |
| Class-aware Selective Loss: Ben-Baruch et al., CVPR 2022 | [arXiv:2110.10955](https://arxiv.org/abs/2110.10955) | LIT | Per-class treatment of unannotated labels. |
| Durand et al., CVPR 2019 (partial labels) | [arXiv:1902.09720](https://arxiv.org/abs/1902.09720) | LIT | Partial BCE. |
| ROLE: Cole et al., CVPR 2021 | [arXiv:2106.09708](https://arxiv.org/abs/2106.09708) | LIT | Single-positive learning. |
| Robinson et al., ICLR 2021, hard-negative contrastive learning | [arXiv:2010.04592](https://arxiv.org/abs/2010.04592) | LIT | Hard-negative weighting. |

## Multiple-instance pooling and glimpses

| Work | Link | Cited in | Used for |
|---|---|---|---|
| WILDCAT (top-k + alpha x bottom-k, alpha 0.7) | not given | POOL, CL, code | `--classification-pooling-bottom-weight`. |
| Pinheiro & Collobert (log-sum-exp pooling, r = 5) | not given | POOL | The temperature arms. |
| HCP / Oquab et al. (hard max pooling) | not given | POOL | The top1 arm. |
| Saccader; GFNet; Wang et al. 2017; RA-CNN | not given | POOL | Classifiers that look at small regions; a ranking hinge. |
| WSDDN: Bilen & Vedaldi, CVPR 2016 | [arXiv:1511.02853](https://arxiv.org/abs/1511.02853) | OV | Its failure mode (the most predictive part wins) matches run 5's. |
| Attention-MIL: Ilse et al. | [arXiv:1802.04712](https://arxiv.org/abs/1802.04712) | OV | Not built. |
| MIST: Ren et al., CVPR 2020 | [arXiv:2004.04725](https://arxiv.org/abs/2004.04725) | OV | Top-p pseudo-boxes; worse than the single peak in experiment 0. |

## Open-vocabulary detection and segmentation

| Work | Link | Cited in | Used for |
|---|---|---|---|
| SAM 3: Carion et al., "SAM 3: Segment Anything with Concepts" | [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) | OV | Presence x region factorization (plan #2); hard negatives. The `facebook/sam3` model made the gold-auto boxes and the missing-labels pass. |
| OWL-ViT: Minderer et al., "Simple Open-Vocabulary Object Detection with Vision Transformers", ECCV 2022 | [arXiv:2205.06230](https://arxiv.org/abs/2205.06230) | OV | |
| OWLv2 / OWL-ST: Minderer, Gritsenko, Houlsby, "Scaling Open-Vocabulary Object Detection", NeurIPS 2023 | [arXiv:2306.09683](https://arxiv.org/abs/2306.09683) | OV | Self-training with pseudo-boxes; mixed mosaic grids (plan #3). The `google/owlv2-base-patch16-ensemble` checkpoint was used as a gold-auto proposer. |
| Grounding DINO: Liu et al. | [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) | OV | `IDEA-Research/grounding-dino-base` as a gold-auto proposer. |
| GLIP: Li et al., "Grounded Language-Image Pre-training", CVPR 2022 | [arXiv:2112.03857](https://arxiv.org/abs/2112.03857) | OV | Tag-conditioned glimpses (not run). |
| Detic: Zhou et al., ECCV 2022 | [arXiv:2201.02605](https://arxiv.org/abs/2201.02605) | OV | Image labels go to the whole image, never to predicted regions. |
| PB-OVD: Gao et al., ECCV 2022 | [arXiv:2111.09452](https://arxiv.org/abs/2111.09452) | OV | Pseudo-boxes from activation maps. |
| SCLIP: Wang et al. | [arXiv:2312.01597](https://arxiv.org/abs/2312.01597) | OV | |
| ClearCLIP: Lan et al., ECCV 2024 | [arXiv:2407.12442](https://arxiv.org/abs/2407.12442) | OV | Why globally trained features carry context. |
| CLIP Surgery: Li et al. | [arXiv:2304.05653](https://arxiv.org/abs/2304.05653) | OV | |
| FILIP: Yao et al., ICLR 2022 | [arXiv:2111.07783](https://arxiv.org/abs/2111.07783) | OV | Patch-max alignment. |
| SPARC: Bica et al., "Improving fine-grained understanding in image-text pre-training", ICML 2024 | [arXiv:2401.09865](https://arxiv.org/abs/2401.09865) | OV | Within-image normalization (not run). |

## Models used as tools (not cited papers)

- **SAM 3** (`facebook/sam3`, transformers `Sam3Model`).
- **OWLv2 base and large.**
- **Grounding DINO base.**
- **Qwen2.5-VL 3B:** a detector and verifier candidate for gold-auto.
- **Claude Opus, Sonnet and Haiku** as box checkers in the gold-auto calibration. Sonnet and Haiku were tried; a
  Sonnet subagent labelled the extended context check.

All of these were used for **evaluation data only**. None trains the tagger: the project's rule is no teacher
models.
