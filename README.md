# MTSN-Spot-ME-MaE

This is the code repository for the submitted paper <b>MTSN: A Multi-Temporal Stream Network for Spotting Facial Macro- and Micro-Expression with Hard and Soft Pseudo-labels </b>.

## How to run the code

<b>Step 1)</b> Installation of packages using pip

``` pip install -r requirements.txt ```

<b>Step 2)</b> Download processed optical flow features from :

<b>The link is hidden at the moment but will be made available soon. </b>
<!--
https://drive.google.com/file/d/1Cn4rux-Hwrt6E1LWO3VL3ddNqOwmgP71/view?usp=sharing
-->
  
<b>Step 3)</b> Place the folder (megc2022-processed-data) accordingly: <br>
>├─megc2022-pretrained-weights <br>
>├─<b>megc2022-processed-data</b> <br>
>├─dataloader.py <br>
>├─network.py <br>
>├─...... <br>
>├─test_main.py <br>
>└─......

<b>Step 4) Evaluation for MEGC 2022 unseen datasets CAS(ME)<sup>3</sup> and SAMM Challenge

``` python test_main.py ```
  
## Additional Notes
  
If you have issue installing torch, run this: <br>
``` pip install torch===1.5.0 torchvision===0.6.0 torchsummary==1.5.1 -f https://download.pytorch.org/whl/torch_stable.html ```
  
##### Please email me at genbing67@gmail.com if you have any inquiries or issues.
