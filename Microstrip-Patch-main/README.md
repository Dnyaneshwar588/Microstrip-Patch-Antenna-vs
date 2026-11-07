📖 Overview

This repository provides a dataset for modeling microstrip patch antennas using machine learning, with a focus on knowledge-based regularization.

The dataset contains mappings between antenna design parameters and their corresponding electromagnetic responses, enabling the development and evaluation of surrogate models for antenna analysis and design.

This dataset accompanies the following publication:

Saçın, E.S., Durgun, A.C., Microstrip Patch Antenna Modeling Using Neural Networks with Knowledge-Based Regularization, Neural Computing and Applications, 37, 3827–3837 (2025).
https://doi.org/10.1007/s00521-024-10860-5

📌 Citation

If you use this dataset, please cite:

@article{sacin2025microstrip,
  title={Microstrip Patch Antenna Modeling Using Neural Networks with Knowledge-Based Regularization},
  author={Saçın, E. S. and Durgun, A. C.},
  journal={Neural Computing and Applications},
  volume={37},
  pages={3827--3837},
  year={2025},
  doi={10.1007/s00521-024-10860-5}
}

📊 Dataset Description

Inputs:The dataset includes key antenna design parameters such as: Patch length and width, Substrate and solder mask thickness, 
material relative permittivity, feed location and configuration. File:input_parameters.xlsx

Outputs: Frequency-dependent S-parameters. File: S11_re.xlsx, S11_im.xlsx

⚙️ Data Generation

Generated using full-wave electromagnetic simulations using Ansys HFSS. The simulation model is provided in the repository.

Note: Detailed simulation settings and methodology are also described in the associated publication.

🎯 Applications

Antenna modeling and prediction, Surrogate modeling, Physics-informed / knowledge-based learning, EM-driven optimization

📬 Contact

For questions or collaborations please contact A.C. Durgun at acdurgun@metu.edu.tr.

Contributions are welcome. Please open an issue to discuss proposed changes or improvements.

🙏 Acknowledgment

If you use this dataset, please cite the associated publication and acknowledge the authors.
