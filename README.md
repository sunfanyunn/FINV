# Partial-View Object View Synthesis via Filtering Inversion 

**3DV 2024 (Highlight)**

Official pytorch implementation

[![Website](doc/badges/badge-website.svg)](https://cs.stanford.edu/~sunfanyun/finv/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2304.00673)


[Fan-Yun Sun](https://sunfanyun.com/), [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay), [Valts Blukis](https://www.cs.cornell.edu/~valts/), [Kevin Lin](https://kevin-thankyou-lin.github.io/), [Danfei Xu](https://faculty.cc.gatech.edu/~danfei/), [Boris Ivanovic](https://www.borisivanovic.com/), [Peter Karkus](http://karkus.tilda.ws/), [Stan Birchfield](https://cecas.clemson.edu/~stb/), [Dieter Fox](https://homes.cs.washington.edu/~fox/), [Ruohan Zhang](https://ai.stanford.edu/~zharu/), [Yunzhu Li](https://yunzhuli.github.io/), [Jiajun Wu](https://jiajunwu.com/), [Marco Pavone](https://research.nvidia.com/person/marco-pavone), [Nick Haber](https://ed.stanford.edu/faculty/nhaber)
<br>

## Usage
1. environmental setup
```
$ pip install -r requirements.txt
```
2. dataset preparation (see below)

3. refer to the sample command below:
```
# Scannet
python main.py scene_id=scene0038_00
```

## Data Preparation
Run `prepare_data.sh`. The resulting directory should have the following file structure:

```
|-- data/
|   |-- scannet/
|   |   |-- scans/
|   |   |-- scan2cad/
|   |   |-- processed_scannet/
|   |-- gt_object_mesh/
```

## TODO
[ ] parallelize the inversion process

## Broader Information
FINV builds upon several previous works:
- [PTI: Pivotal Tuning for Latent-based editing of Real Images](https://github.com/danielroich/PTI)
- [Efficient Geometry-aware 3D Generative Adversarial Networks](https://github.com/NVlabs/eg3d)
- [GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images](https://github.com/nv-tlabs/GET3D)


## Citation
```latex
@article{sun2023partial,
  title={Partial-View Object View Synthesis via Filtered Inversion},
  author={Sun, Fan-Yun and Tremblay, Jonathan and Blukis, Valts and Lin, Kevin and Xu, Danfei and Ivanovic, Boris and Karkus, Peter and Birchfield, Stan and Fox, Dieter and Zhang, Ruohan and others},
  journal={International Conference on 3D Vision (3DV)},
  year={2024}
}
`````
