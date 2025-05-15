# Mg-Corrision prediction model for the AI<sup>2</sup> project

The AI<sup>2</sup> project investigates the potential of different chemical Mg corrosion inhibitors.
To determine the inhibitive effects of different inhibitors multiple corrions experiments have been performed.
After every experiment the corroded residues have to removed by way of cleaning the sample with chromic acid to determine a heatmap of Mg volume loss.
This repository contains the code for a computer vision model that predicts the volume loss heatmap from the corroded sample profilometer images.

## Architecture
To predict the heatmap from the sample images a model is trained that is based on a [U-Net structure](https://doi.org/10.1007/978-3-319-24574-4_28) as commonly used for image segmentation.
