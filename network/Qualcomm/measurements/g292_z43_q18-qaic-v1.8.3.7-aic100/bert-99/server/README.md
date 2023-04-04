# Qualcomm Cloud AI 100 - Network - BERT-99 Server

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

## Server

Note that when tuning the Server target QPS one can use e.g. `--query_count=1000000`: ~75 seconds should be enough to establish whether we pass the latency constraint.

### Launch the server program (`pf003`)

#### BERT-99

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99 --vc=16 --sdk=1.8.3.7 \
--mode=performance --scenario=server --server_target_qps=1234 --server_override_batch_size=200 \
--sut=g292_z43_q18 --pre_fan=250 --post_fan=100 --network_server --verbose --container=$CONTAINER_ID
```

### Run the client program (`pf002`)

#### BERT-99

##### Accuracy

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99 --sdk=1.8.3.7 \
--mode=accuracy --scenario=server --server_target_qps=13000 \
--sut=g292_z43_q18 --network_client --ecc=off --verbose --container=$CONTAINER_ID
```

##### Performance

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99 --sdk=1.8.3.7 \
--mode=performance --scenario=server --server_target_qps=13000 \
--sut=g292_z43_q18 --network_client --ecc=off --verbose --container=$CONTAINER_ID
```

##### Compliance

```
ck run cmdgen:benchmark.packed-bert.qaic-loadgen --model=bert-99 --sdk=1.8.3.7 \
--compliance,=TEST01,TEST05 --scenario=server --server_target_qps=13000 \
--sut=g292_z43_q18 --network_client --ecc=off --verbose --container=$CONTAINER_ID
```
