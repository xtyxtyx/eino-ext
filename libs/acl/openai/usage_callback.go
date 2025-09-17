/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package openai

import (
	"context"

	"github.com/cloudwego/eino/schema"
	goopenai "github.com/meguminnnnnnnnn/go-openai"
)

// ExtendedTokenUsage represents usage information in OpenRouter format
// Reference: https://openrouter.ai/docs/use-cases/usage-accounting
type ExtendedTokenUsage struct {
	// Basic token counts
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// Extended details for prompt tokens
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`

	// Extended details for completion tokens
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`

	// Cost information
	Cost        *float64     `json:"cost,omitempty"`         // Total cost in credits
	CostDetails *CostDetails `json:"cost_details,omitempty"` // Detailed cost breakdown
}

// PromptTokensDetails contains detailed breakdown of prompt tokens
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"` // Number of tokens read from cache
	AudioTokens  int `json:"audio_tokens,omitempty"`  // Number of audio tokens
}

// CompletionTokensDetails contains detailed breakdown of completion tokens
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"` // Number of reasoning tokens (for o1 models)
}

// CostDetails contains detailed cost breakdown
type CostDetails struct {
	UpstreamInferenceCost *float64 `json:"upstream_inference_cost,omitempty"` // Cost charged by upstream provider (BYOK only)
}

// UsageCallbackHandler defines the interface for handling usage information
type UsageCallbackHandler interface {
	// OnUsage is called when usage information is available
	OnUsage(ctx context.Context, usage *ExtendedTokenUsage) error
}

// UsageCallbackFunc is a function type that implements UsageCallbackHandler
type UsageCallbackFunc func(ctx context.Context, usage *ExtendedTokenUsage) error

// OnUsage implements UsageCallbackHandler
func (f UsageCallbackFunc) OnUsage(ctx context.Context, usage *ExtendedTokenUsage) error {
	return f(ctx, usage)
}

// UsageCallbackConfig configures usage callback behavior
type UsageCallbackConfig struct {
	// Enabled determines if usage callbacks are active
	Enabled bool

	// Handler is the callback handler to invoke
	Handler UsageCallbackHandler

	// IncludeUsageInRequest enables sending usage.include=true in API requests
	// This tells the API to include usage information in responses
	IncludeUsageInRequest bool
}

// convertToExtendedUsage converts standard schema.TokenUsage to ExtendedTokenUsage
// rawUsage should be the original openai.Usage object which may contain extended fields
func convertToExtendedUsage(usage *schema.TokenUsage, rawUsage interface{}) *ExtendedTokenUsage {
	if usage == nil {
		return nil
	}

	extended := &ExtendedTokenUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}

	// Try to extract extended usage information from the raw openai.Usage struct
	// This supports OpenRouter and other providers that include cost information
	if rawUsage != nil {
		extractExtendedUsageFields(extended, rawUsage)
	}

	return extended
}

// extractExtendedUsageFields extracts cost and other extended fields from openai.Usage
func extractExtendedUsageFields(extended *ExtendedTokenUsage, rawUsage interface{}) {
	// Type assert to openai.Usage to get the enhanced fields
	if openaiUsage, ok := rawUsage.(*goopenai.Usage); ok {
		// Extract cost information directly from the struct
		if openaiUsage.Cost != nil {
			extended.Cost = openaiUsage.Cost
		}

		// Extract cost details
		if openaiUsage.CostDetails != nil {
			extended.CostDetails = &CostDetails{
				UpstreamInferenceCost: openaiUsage.CostDetails.UpstreamInferenceCost,
			}
		}

		// Extract prompt token details
		if openaiUsage.PromptTokensDetails != nil {
			extended.PromptTokensDetails = &PromptTokensDetails{
				CachedTokens: openaiUsage.PromptTokensDetails.CachedTokens,
				AudioTokens:  openaiUsage.PromptTokensDetails.AudioTokens,
			}
		}

		// Extract completion token details
		if openaiUsage.CompletionTokensDetails != nil {
			extended.CompletionTokensDetails = &CompletionTokensDetails{
				ReasoningTokens: openaiUsage.CompletionTokensDetails.ReasoningTokens,
			}
		}
	}
}

// triggerUsageCallback triggers the usage callback if configured
func (c *Client) triggerUsageCallback(ctx context.Context, usage *schema.TokenUsage, rawUsage interface{}) {
	if c.usageCallbackConfig == nil || !c.usageCallbackConfig.Enabled || c.usageCallbackConfig.Handler == nil {
		return
	}

	extendedUsage := convertToExtendedUsage(usage, rawUsage)
	if extendedUsage == nil {
		return
	}

	// Call the usage callback handler
	// We don't want callback errors to affect the main flow, so we just log them
	if err := c.usageCallbackConfig.Handler.OnUsage(ctx, extendedUsage); err != nil {
		// In a real implementation, we might want to log this error
		// For now, we silently ignore it to not break the main flow
		_ = err
	}
}
