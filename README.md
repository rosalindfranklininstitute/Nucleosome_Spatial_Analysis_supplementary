# Nucleosome Spatial Analysis supplementary

Source code generated using the aid of Anthropic's Claude Sonnet 5 and OpenAI's ChatGPT GPT-5 LLMs.

## Installation

In a Python 3.11 (or never) environment you can install the package by running the following commands:

```
pip install -e .
```

(Approximate time for installation: 1 minute and 10-20 seconds)

## Command-line interface

There are 1 commands available:
* `voronoi`

And one script available:
* `unionise`

## Demo

To demo the creation of feature descriptors run:

### For Voronoi (run time approximately 1 min to 40 sec)
```shell
voronoi --group inactive "$(pwd)/demo_data/demo_inactive*.star" \
  --group active "$(pwd)/demo_data/demo_active*.star" \
  --angpix 2.33 \
  --full-x 4096 \
  --full-y 4096 \
  --full-z 1200 \
  --crop-x 3600 \
  --crop-y 3600 \
  --crop-z 200 \
  --out-prefix demo_voronoi \
  --out-dir $(pwd)/demo_outputs/
```
Expected outputs: 
- `demo_outputs/demo_voronoi_CDF_nm3.svg`

### For Unionise (run time approximately 1 minute 10-20 seconds )
```shell
cd path/to/Nucleosome_Spatial_Analysis_supplementary
python src/unionise.py
```

If you want to use the unionise script for other data visit lines 168-172 of the
`src/unionise.py` and edit the paths to the appropriate directories
   
Expected outputs: 
- `demo_unionise_outputs/J271-3_TS_03_union_ellipsoid_ax12.0_rad24.0_cg27.0_ov0.1.star`

## License

   Copyright [2026] [Rosalind Franklin Institute]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
