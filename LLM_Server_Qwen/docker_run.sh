#!/bin/bash

cd /workspace/LLM_Server_Qwen && nohup python manage_llm_server_qwen.py >/dev/null 2>&1 &
echo "server runing"
/bin/bash
