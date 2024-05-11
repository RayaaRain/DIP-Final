### Execution

```bash
python main.py [--Options OpArgs]
```

Options:
- `--input_path PATH`: Path to input image. Default to `./SampleImages/lena.png`  
- `--output_path PATH`: Path to output image. Default to `./OutputImages/edge_map.png`  
- `-tl INT`, `--threshold_low INT`: threshold for discard(unit: pixels). Default to 8  
- `-th FLOAT`, `--threshold_high FLOAT`: threshold for proportion of edge(range: 0~1). If not None, override default value.   
- `--save_partial`: save the image during curvature prediction. Default to false. Turn on would slow down the process.
- `--save_anchor`: save the anchor map.