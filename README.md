# Encod ECG signals into images
Run the following python script to convert ECG signals into images

```python ecg_extract_lead_image.py --style=grid --method=gaf,rp,mtf --output_path=data/output/abnormalPTB-XL_Grid```

You can modify the visualization style by using the ```--style``` argument. The valid options for this argument are ```concat``` and ```grid```.

Moreover, you can customize the image visualization approach by adjusting the ```--methods``` argument. Make sure to separate multiple methods with a comma, for example ```gaf,rp,mtf```
