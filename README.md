# Partial-View Object View Synthesis via Filtering Inversion

[Fan-Yun Sun](https://sunfanyun.com/), [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay), Valts Blukis, Kevin Lin, Danfei Xu, Boris Ivanovic, Peter Karkus, Stan Birchfield, Dieter Fox, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Marco Pavone, Nick Haber

[Project Page](https://cs.stanford.edu/~sunfanyun/finv/)

## Data Preparation
Run `prepare_data.sh`. The resulting directory structure should look like:

```
|-- data/
|   |-- scannet/
|   |   |-- scans/
|   |   |-- scan2cad/
|   |   |-- processed_scannet/
|   |-- gt_object_mesh/
```

## Broader Information

FINV builds upon several previous works:

- [PTI: Pivotal Tuning for Latent-based editing of Real Images](https://github.com/danielroich/PTI)
- [Efficient Geometry-aware 3D Generative Adversarial Networks](https://github.com/NVlabs/eg3d)
- [GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images](https://github.com/nv-tlabs/GET3D)

## Run

refer to `go.sh`


## Citation

```latex
@article{sun2023partial,
  title={Partial-View Object View Synthesis via Filtered Inversion},
  author={Sun, Fan-Yun and Tremblay, Jonathan and Blukis, Valts and Lin, Kevin and Xu, Danfei and Ivanovic, Boris and Karkus, Peter and Birchfield, Stan and Fox, Dieter and Zhang, Ruohan and others},
  journal={arXiv preprint arXiv:2304.00673},
  year={2023}
}
`````

