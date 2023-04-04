# How to download DLRM model to NEUCHIPS RecAccel PCIe cards

## 1. Download dlrm model data from remote storage
   wget https://zenodo.org/record/7623702/files/n3000.dlrm.tar.gz?download=1

## 2. Create ./model_data folder.
`   $mkdir model_data`

## 3. Extract the .gz file to ./model_data
`tar xzf n3000.dlrm.tar.gz -C ./model_data/`

## 4. Download to RecAccel PCIe cards.
   if use single card,
`       $./n3000_dl 0`

   if use 8 cards,
```bash
       $./n3000_dl 0  

       $./n3000_dl 1  

       $./n3000_dl 2  

       $./n3000_dl 3  

       $./n3000_dl 4  

       $./n3000_dl 5  

       $./n3000_dl 6  

       $./n3000_dl 7  

