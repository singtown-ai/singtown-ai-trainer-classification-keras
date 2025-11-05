# SingTown AI Trainer Classification Keras

## Support Models

- mobilenet_v1_0.25_128
- mobilenet_v1_0.25_160
- mobilenet_v1_0.25_192
- mobilenet_v1_0.25_224
- mobilenet_v1_0.50_128
- mobilenet_v1_0.50_160
- mobilenet_v1_0.50_192
- mobilenet_v1_0.50_224
- mobilenet_v1_0.75_128
- mobilenet_v1_0.75_160
- mobilenet_v1_0.75_192
- mobilenet_v1_0.75_224
- mobilenet_v1_1.0_128
- mobilenet_v1_1.0_160
- mobilenet_v1_1.0_192
- mobilenet_v1_1.0_224
- mobilenet_v2_0.35_96
- mobilenet_v2_0.35_128
- mobilenet_v2_0.35_160
- mobilenet_v2_0.35_192
- mobilenet_v2_0.35_224
- mobilenet_v2_0.50_96
- mobilenet_v2_0.50_128
- mobilenet_v2_0.50_160
- mobilenet_v2_0.50_192
- mobilenet_v2_0.50_224
- mobilenet_v2_0.75_96
- mobilenet_v2_0.75_128
- mobilenet_v2_0.75_160
- mobilenet_v2_0.75_192
- mobilenet_v2_0.75_224
- mobilenet_v2_1.0_96
- mobilenet_v2_1.0_128
- mobilenet_v2_1.0_160
- mobilenet_v2_1.0_192
- mobilenet_v2_1.0_224
- mobilenet_v2_1.3_224
- mobilenet_v2_1.4_224

## Test

```
# test
unset SINGTOWN_AI_HOST
unset SINGTOWN_AI_TOKEN
unset SINGTOWN_AI_TASK_ID
export SINGTOWN_AI_MOCK_TASK_PATH="../mock-task.json"
export SINGTOWN_AI_MOCK_DATASET_PATH="../classification-20.json"
uv run main.py
```


```
# test
export SINGTOWN_AI_HOST="https://ai.singtown.com"
export SINGTOWN_AI_TOKEN="your token"
export SINGTOWN_AI_TASK_ID="your id"
unset SINGTOWN_AI_MOCK_TASK_PATH
unset SINGTOWN_AI_MOCK_DATASET_PATH
uv run main.py
```
