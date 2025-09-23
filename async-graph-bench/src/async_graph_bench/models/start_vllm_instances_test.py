from start_vllm_instances import start_vllm_instances
import asyncio


async def main():
    ports = await start_vllm_instances(
        available_gpus=[0, 1, 2, 3, 4, 5, 6, 7],
        gpus_per_model=1,
        model="mistralai/Ministral-8B-Instruct-2410",
        model_params={
            "tensor_parallel_size": 1,
            "tokenizer_mode": "mistral"
        },
        poll_timeout=600,  # 10 mins
    )
    print("All vllm instances ready:", ports)


if __name__ == "__main__":
    asyncio.run(main())
