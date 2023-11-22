# ECG_image_encoding

## [Automated Cardiovascular Record Retrieval by Multimodal Learning between Electrocardiogram and Clinical Report]([https://arxiv.org/abs/2212.08044](https://arxiv.org/abs/2304.06286))

Run the following Python script to convert ECG signals into images

```python ecg_extract_lead_image.py --style=grid --method=gaf,rp,mtf --output_path=data/output/abnormalPTB-XL_Grid```

You can modify the visualization style by using the ```--style``` argument. The valid options for this argument are ```concat``` and ```grid```.

Moreover, you can customize the image visualization approach by adjusting the ```--methods``` argument. Make sure to separate multiple methods with a comma, for example ```gaf,rp,mtf```

## Citation

If you feel our code or models help in your research, kindly cite our papers:

```
@inproceedings{Qiu2023AutomatedCR,
  title={Automated Cardiovascular Record Retrieval by Multimodal Learning between Electrocardiogram and Clinical Report},
  author={Jielin Qiu and Jiacheng Zhu and Shiqi Liu and William Jongwon Han and Jingqi Zhang and Chaojing Duan and Michael Rosenberg and Emerson Liu and Douglas Weber and Ding Zhao},
  journal={Proceedings of the 3nd Machine Learning for Health symposium, PMLR},
  year={2023},
}

@article{qiu2023converting,
  title={Converting ECG Signals to Images for Efficient Image-text Retrieval via Encoding},
  author={Qiu, Jielin and Zhu, Jiacheng and Liu, Shiqi and Han, William and Zhang, Jingqi and Duan, Chaojing and Rosenberg, Michael and Liu, Emerson and Weber, Douglas and Zhao, Ding},
  journal={arXiv preprint arXiv:2304.06286},
  year={2023}
}

```
