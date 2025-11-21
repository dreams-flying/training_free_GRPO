# Rate Limit Handling Guide

## Overview

When running Training-Free GRPO on Windows (or any platform), you may encounter API rate limit errors:

```
openai.RateLimitError: local_rate_limited
```

This guide explains how to handle and avoid these errors.

## What Causes Rate Limit Errors?

1. **Too many concurrent requests**: The default `--rollout_concurrency` might be too high for your API tier
2. **API quota exceeded**: Your API key has limited requests per minute/hour
3. **Insufficient cooling period**: Not enough delay between batches of requests

## Solutions

### 1. Reduce Concurrency (Recommended)

Lower the `--rollout_concurrency` parameter when running training or evaluation:

**For Training:**
```bash
python training_free_grpo/train.py \
    --mode agent \
    --domain math \
    --experiment_name test \
    --dataset DAPO-Math-17k \
    --dataset_truncate 100 \
    --rollout_concurrency 5  # Reduced from default 128
```

**For Evaluation:**
```bash
python training_free_grpo/main.py \
    --mode agent \
    --domain math \
    --experiment_name test \
    --dataset AIME24 \
    --rollout_concurrency 5  # Reduced from default 128
```

### 2. Recommended Concurrency Settings by API Tier

- **Free tier / DeepSeek free API**: `--rollout_concurrency 2-5`
- **Basic tier**: `--rollout_concurrency 10-20`
- **Pro tier**: `--rollout_concurrency 50-100`
- **Enterprise tier**: `--rollout_concurrency 100-200`

### 3. Built-in Retry Mechanism

The code now includes automatic retry with exponential backoff:
- **Retry attempts**: 5 (default)
- **Backoff strategy**: 10s → 20s → 40s → 80s → 160s
- **Handles**: Rate limit errors, API errors, network errors

### 4. Monitor Your API Usage

Check your API provider's dashboard to:
- View current rate limits
- Monitor request quota
- Upgrade tier if needed

### 5. Optimize Batch Size

For training, also consider reducing `--batchsize`:

```bash
python training_free_grpo/train.py \
    --mode agent \
    --domain math \
    --experiment_name test \
    --dataset DAPO-Math-17k \
    --dataset_truncate 100 \
    --batchsize 4 \              # Reduced batch size
    --rollout_concurrency 5      # Reduced concurrency
```

## Example: Conservative Settings for Windows

If you frequently encounter rate limits, use these conservative settings:

```bash
# Training with conservative settings
python training_free_grpo/train.py \
    --mode agent \
    --domain math \
    --experiment_name conservative_test \
    --dataset DAPO-Math-17k \
    --dataset_truncate 50 \
    --epochs 2 \
    --batchsize 10 \
    --grpo_n 3 \
    --rollout_concurrency 3 \
    --rollout_temperature 0.7 \
    --task_timeout 1800

# Evaluation with conservative settings
python training_free_grpo/main.py \
    --mode agent \
    --domain math \
    --experiment_name conservative_eval \
    --dataset AIME24 \
    --rollout_concurrency 3 \
    --pass_k 5
```

## Understanding the Error Messages

When rate limits occur, you'll see messages like:

```
Rate limit reached: Error code: 429 - {'error': {...}}. Retrying in 10s... (attempt 1/5)
Rate limit reached: Error code: 429 - {'error': {...}}. Retrying in 20s... (attempt 2/5)
```

This is **normal behavior** - the system will automatically retry with increasing delays.

## API Provider Specific Tips

### DeepSeek API
- Free tier: Very limited, use `--rollout_concurrency 2`
- Rate limit: ~60 requests per minute
- Consider upgrading to paid tier for better performance

### OpenAI API
- Check your rate limits at: https://platform.openai.com/account/limits
- GPT-4: Higher limits but more expensive
- GPT-3.5: Lower cost, higher rate limits

### Custom API Endpoints
- Consult your provider's documentation for rate limits
- Adjust concurrency accordingly

## Troubleshooting

**Problem**: Still getting rate limit errors even with low concurrency

**Solutions**:
1. Wait a few minutes and try again (quota may reset)
2. Check if multiple processes are using the same API key
3. Verify your API tier and upgrade if needed
4. Contact your API provider support

**Problem**: Training/evaluation is too slow with low concurrency

**Solutions**:
1. Upgrade your API tier for higher rate limits
2. Use a faster model (e.g., gpt-3.5-turbo instead of gpt-4)
3. Run smaller batches and combine results
4. Consider using multiple API keys (if allowed by provider)

## Best Practices

1. **Start small**: Begin with low concurrency and gradually increase
2. **Monitor logs**: Watch for rate limit warnings
3. **Use checkpoints**: The code saves progress, so you can resume if interrupted
4. **Plan for delays**: Factor in retry times when estimating completion time
5. **Check quotas**: Ensure you have sufficient API quota before starting large experiments

## Need Help?

If you continue to experience issues:
1. Check your `.env` file configuration
2. Verify API key is valid and active
3. Review your API provider's rate limit documentation
4. Open an issue on GitHub with error details
