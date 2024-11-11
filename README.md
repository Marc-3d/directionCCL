### Motivation
My goal is to adapt the standard connected components labeling (CCL) algorithm to work on vector fields. Unlike the standard CCL, which connects neighboring pixels with value the same value in a binary mask, this "direction-based CCL" or "directionCCL" for short, connects neighboring pixels in a vector field if the angle between neighboring vectors is smaller than a certain threshold. 

### Implementation
I have taken the code from `ImageComponentAnalysis.jl` and I have modified it a bit to (A) accept vector fields as an input and (B) include the dot product between vectors as the "grouping criterium".

### Examples
I came up with the idea for direction-based CCL for the purpose of post-processing 2D+t and 3D+t vector fields obtained with particle image velocimetry (PIV). However, directionCCL can also be applied to gradients in an image. Gradients are vector fields that usually point towards edges in an image, so my expectation was that this algorithm would be able to segment the pixels belonging to smooth contours in an image. 

#### Example 1: Outlining the contour of organoids in 2D