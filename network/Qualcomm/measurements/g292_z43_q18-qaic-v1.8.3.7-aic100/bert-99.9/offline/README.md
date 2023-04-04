# Qualcomm Cloud AI 100 - Network - BERT-99.9 Offline

# Launch Docker containers (common to BERT-99/BERT-99.9, Offline/Server)

## Launch a Docker container on the server side (`pf003`)

```
CONTAINER_ID=$(ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model_name=bert --sdk=1.8.3.7 \
--docker=container_only --docker_os=ubuntu --docker_port --network_server_port=7276 --out=none)
```

<details><pre>
$ docker container ps
CONTAINER ID   IMAGE                             COMMAND               CREATED         STATUS         PORTS                                       NAMES
ec296f9afeea   krai/mlperf.bert:ubuntu_1.8.3.7   "/bin/bash -c bash"   2 minutes ago   Up 2 minutes   0.0.0.0:7276->7276/tcp, :::7276->7276/tcp   musing_mclean
</pre></details>

## Launch a Docker container on the client side (`pf002`)

```
CONTAINER_ID=$(ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model_name=bert --sdk=1.8.3.7 \
--docker=container_only --docker_os=ubuntu --experiment_dir --out=none)
```

<details><pre>
$ docker container ps
CONTAINER ID   IMAGE                             COMMAND               CREATED          STATUS          PORTS     NAMES
0b13f209b79e   krai/mlperf.bert:ubuntu_1.8.3.7   "/bin/bash -c bash"   13 seconds ago   Up 12 seconds             hungry_mcclintock
</pre></details>


# Scenarios

As usual, we use different values for `--target_qps`: higher for Offline (e.g. `--offline_target_qps=14000`), lower for Server (e.g. `--server_target_qps=13000`).

We also use different values for `--override_batch_size`: higher for Offline (e.g.`--offline_override_batch_size=1000`), lower for Server (e.g. `--server_override_batch_size=100`).


## Offline

### Launch the server program (`pf003`)

Note that we use different `--vc` settings for different models: `--vc=16` for BERT-99; `--vc=13` for BERT-99.9.

We set `--pre_fan=250 --post_fan=150` as usual for `--sut=g292_z43_q18`. We use a dummy mandatory parameter `--offline_target_qps=1234`.

We kill the server program (`sudo pkill bert-server`) after all runs for a particular scenario and model.


#### BERT-99.9

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99.9 --vc=13 --sdk=1.8.3.7 \
--mode=performance --scenario=offline --offline_target_qps=1234 --offline_override_batch_size=1000 \
--sut=g292_z43_q18 --pre_fan=250 --post_fan=150 --network_server --verbose --container=$CONTAINER_ID
```

### Run the client program (`pf002`)

Note that we run with `--network_client --ecc=off` on the client side, as there
is no need to set ECC on accelerators (even if they are present). It is also
optional to use `--pre_fan=150 --post_fan=100`, with little work being done on
the client side.

#### BERT-99.9

##### Accuracy

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99.9 --sdk=1.8.3.7 \
--mode=accuracy --scenario=offline --offline_target_qps=7000 \
--sut=g292_z43_q18 --network_client --ecc=off --verbose --container=$CONTAINER_ID
```

##### Performance

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99.9 --sdk=1.8.3.7 \
--mode=performance --scenario=offline --offline_target_qps=7000 \
--sut=g292_z43_q18 --network_client --ecc=off --verbose --container=$CONTAINER_ID
```

##### Compliance

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99.9 --sdk=1.8.3.7 \
--compliance,=TEST01,TEST05 --scenario=offline --offline_target_qps=7000 \
--sut=g292_z43_q18 --network_client --ecc=off --verbose --container=$CONTAINER_ID
```
