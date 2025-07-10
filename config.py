from copy import deepcopy

base_config = {
    "generation_model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "is_chat_model": True,
    "instruct_tokens": ("[INST]","[/INST]"),
    "index_builder": {
        "tokenizer_model_name": None,
        "chunk_size": 200,
        "overlap": 0,
        "passes": 10,
        },
    "ralm": {
        "top_k_docs": 2,
        "system_prompt": "You are a truthful expert question-answering bot and should correctly and concisely answer the following question",
        "repeat_system_prompt": True,
        "stride": -1,
        "query_len": 200,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.1,
        "num_beams": 2,
        "max_new_tokens": 25,
        "batch_size": 8,
        "generated_questions": False
        }
    }
base_config["index_builder"]["tokenizer_model_name"] = base_config["generation_model_name"]

# Generate configurations for different runs
def generate_configurations(base_config, configs):
    complete_configs = {}
    # Copy the base config and update with the specific values
    for key, config_values in configs.items():
        config = deepcopy(base_config)
        for config_key, value in config_values.items():
            if isinstance(value, dict):
                config[config_key].update(value)
            else:
                config[config_key] = value
        complete_configs[key] = config
    return complete_configs

configs_run = {
    "Generated_Questions": {
    "ralm": {
        "top_k_docs": 1,
        "generated_questions": True
        }
    },
    "Document_k2": {
    "ralm": {
        "top_k_docs": 2,
        }
    },
    "Document_k4": {
    "ralm": {
        "top_k_docs": 2,
        }
    },
    "Document_k8": {
    "ralm": {
        "top_k_docs": 8,
        }
    },
    "Document_k16": {
    "ralm": {
        "top_k_docs": 16,
        }
    },
    "Document_k32": {
    "ralm": {
        "top_k_docs": 32,
        }
    }
}

configs_run = generate_configurations(base_config, configs_run)