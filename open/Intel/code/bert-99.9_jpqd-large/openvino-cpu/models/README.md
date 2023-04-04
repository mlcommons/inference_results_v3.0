# JPQD Models
Models were split using:
```bash
find * -type f -size +50M -exec split -b 50M -d {} {}_p \;
find * -type f -size +50M > large_model_files.txt
find * -type f -size +50M -delete
```

To retrieve full file, run the following from [code/bert-99.9_jpqd-large/openvino-cpu/](/open/Intel/code/bert-99.9_jpqd-large/openvino-cpu/):
```bash
./prepare_models.sh
```
