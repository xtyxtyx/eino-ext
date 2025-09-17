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
	"testing"

	"github.com/cloudwego/eino/schema"
	goopenai "github.com/meguminnnnnnnnn/go-openai"
)

func TestUsageCallbackFunc(t *testing.T) {
	called := false
	var receivedUsage *ExtendedTokenUsage

	callback := UsageCallbackFunc(func(ctx context.Context, usage *ExtendedTokenUsage) error {
		called = true
		receivedUsage = usage
		return nil
	})

	cost := 0.01
	usage := &ExtendedTokenUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
		Cost:             &cost,
	}

	err := callback.OnUsage(context.Background(), usage)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if !called {
		t.Error("Expected callback to be called")
	}

	if receivedUsage == nil {
		t.Error("Expected to receive usage data")
	} else {
		if receivedUsage.PromptTokens != 10 {
			t.Errorf("Expected PromptTokens=10, got=%d", receivedUsage.PromptTokens)
		}
		if receivedUsage.CompletionTokens != 20 {
			t.Errorf("Expected CompletionTokens=20, got=%d", receivedUsage.CompletionTokens)
		}
		if receivedUsage.TotalTokens != 30 {
			t.Errorf("Expected TotalTokens=30, got=%d", receivedUsage.TotalTokens)
		}
		if receivedUsage.Cost == nil || *receivedUsage.Cost != 0.01 {
			if receivedUsage.Cost == nil {
				t.Error("Expected Cost to be set, got nil")
			} else {
				t.Errorf("Expected Cost=0.01, got=%f", *receivedUsage.Cost)
			}
		}
	}
}

func TestConvertToExtendedUsage(t *testing.T) {
	// Test with nil usage
	result := convertToExtendedUsage(nil, nil)
	if result != nil {
		t.Error("Expected nil result for nil usage")
	}

	// Test basic conversion
	usage := &schema.TokenUsage{
		PromptTokens:     15,
		CompletionTokens: 25,
		TotalTokens:      40,
	}

	result = convertToExtendedUsage(usage, nil)
	if result == nil {
		t.Fatal("Expected non-nil result")
	}

	if result.PromptTokens != 15 {
		t.Errorf("Expected PromptTokens=15, got=%d", result.PromptTokens)
	}
	if result.CompletionTokens != 25 {
		t.Errorf("Expected CompletionTokens=25, got=%d", result.CompletionTokens)
	}
	if result.TotalTokens != 40 {
		t.Errorf("Expected TotalTokens=40, got=%d", result.TotalTokens)
	}
	if result.Cost != nil {
		t.Errorf("Expected Cost to be nil without config, got=%f", *result.Cost)
	}
}

func TestConvertToExtendedUsageWithRawUsage(t *testing.T) {
	usage := &schema.TokenUsage{
		PromptTokens:     1000,
		CompletionTokens: 500,
		TotalTokens:      1500,
	}

	// Mock go-openai Usage struct with cost information (like from OpenRouter)
	cost := 2.5
	upstreamCost := 2.0
	rawUsage := &goopenai.Usage{
		PromptTokens:     1000,
		CompletionTokens: 500,
		TotalTokens:      1500,
		Cost:             &cost,
		CostDetails: &goopenai.CostDetails{
			UpstreamInferenceCost: &upstreamCost,
		},
		PromptTokensDetails: &goopenai.PromptTokensDetails{
			CachedTokens: 100,
			AudioTokens:  50,
		},
		CompletionTokensDetails: &goopenai.CompletionTokensDetails{
			ReasoningTokens: 25,
		},
	}

	result := convertToExtendedUsage(usage, rawUsage)
	if result == nil {
		t.Fatal("Expected non-nil result")
	}

	// Check cost extraction
	if result.Cost == nil || *result.Cost != 2.5 {
		if result.Cost == nil {
			t.Error("Expected Cost to be set, got nil")
		} else {
			t.Errorf("Expected Cost=2.5, got=%f", *result.Cost)
		}
	}

	// Check cost details
	if result.CostDetails == nil {
		t.Error("Expected CostDetails to be set")
	} else {
		if result.CostDetails.UpstreamInferenceCost == nil || *result.CostDetails.UpstreamInferenceCost != 2.0 {
			if result.CostDetails.UpstreamInferenceCost == nil {
				t.Error("Expected UpstreamInferenceCost to be set, got nil")
			} else {
				t.Errorf("Expected UpstreamInferenceCost=2.0, got=%f", *result.CostDetails.UpstreamInferenceCost)
			}
		}
	}

	// Check prompt token details
	if result.PromptTokensDetails == nil {
		t.Error("Expected PromptTokensDetails to be set")
	} else {
		if result.PromptTokensDetails.CachedTokens != 100 {
			t.Errorf("Expected CachedTokens=100, got=%d", result.PromptTokensDetails.CachedTokens)
		}
		if result.PromptTokensDetails.AudioTokens != 50 {
			t.Errorf("Expected AudioTokens=50, got=%d", result.PromptTokensDetails.AudioTokens)
		}
	}

	// Check completion token details
	if result.CompletionTokensDetails == nil {
		t.Error("Expected CompletionTokensDetails to be set")
	} else {
		if result.CompletionTokensDetails.ReasoningTokens != 25 {
			t.Errorf("Expected ReasoningTokens=25, got=%d", result.CompletionTokensDetails.ReasoningTokens)
		}
	}
}

func TestClientWithUsageCallback(t *testing.T) {
	// Create a mock client
	client := &Client{
		usageCallbackConfig: nil,
	}

	// Test setting usage callback
	callback := UsageCallbackFunc(func(ctx context.Context, usage *ExtendedTokenUsage) error {
		return nil
	})

	config := &UsageCallbackConfig{
		Enabled: true,
		Handler: callback,
	}

	// Test WithUsageCallback
	newClient := client.WithUsageCallback(config)
	if newClient == client {
		t.Error("Expected new client instance")
	}
	if newClient.usageCallbackConfig != config {
		t.Error("Expected usage callback config to be set")
	}

	// Test SetUsageCallback
	client.SetUsageCallback(config)
	if client.usageCallbackConfig != config {
		t.Error("Expected usage callback config to be set on original client")
	}
}

func TestTriggerUsageCallback(t *testing.T) {
	called := false
	var receivedUsage *ExtendedTokenUsage

	callback := UsageCallbackFunc(func(ctx context.Context, usage *ExtendedTokenUsage) error {
		called = true
		receivedUsage = usage
		return nil
	})

	config := &UsageCallbackConfig{
		Enabled: true,
		Handler: callback,
	}

	client := &Client{
		usageCallbackConfig: config,
	}

	usage := &schema.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 200,
		TotalTokens:      300,
	}

	// Trigger the callback
	client.triggerUsageCallback(context.Background(), usage, nil)

	if !called {
		t.Error("Expected callback to be called")
	}

	if receivedUsage == nil {
		t.Error("Expected to receive usage data")
	} else {
		if receivedUsage.PromptTokens != 100 {
			t.Errorf("Expected PromptTokens=100, got=%d", receivedUsage.PromptTokens)
		}
		if receivedUsage.CompletionTokens != 200 {
			t.Errorf("Expected CompletionTokens=200, got=%d", receivedUsage.CompletionTokens)
		}
		if receivedUsage.TotalTokens != 300 {
			t.Errorf("Expected TotalTokens=300, got=%d", receivedUsage.TotalTokens)
		}
	}
}

func TestTriggerUsageCallbackDisabled(t *testing.T) {
	called := false

	callback := UsageCallbackFunc(func(ctx context.Context, usage *ExtendedTokenUsage) error {
		called = true
		return nil
	})

	config := &UsageCallbackConfig{
		Enabled: false, // Disabled
		Handler: callback,
	}

	client := &Client{
		usageCallbackConfig: config,
	}

	usage := &schema.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 200,
		TotalTokens:      300,
	}

	// Trigger the callback
	client.triggerUsageCallback(context.Background(), usage, nil)

	if called {
		t.Error("Expected callback NOT to be called when disabled")
	}
}

func TestTriggerUsageCallbackNoConfig(t *testing.T) {
	client := &Client{
		usageCallbackConfig: nil, // No config
	}

	usage := &schema.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 200,
		TotalTokens:      300,
	}

	// This should not panic
	client.triggerUsageCallback(context.Background(), usage, nil)
}

func TestIncludeUsageInRequest(t *testing.T) {
	config := &UsageCallbackConfig{
		Enabled:               true,
		IncludeUsageInRequest: true,
		Handler: UsageCallbackFunc(func(ctx context.Context, usage *ExtendedTokenUsage) error {
			return nil
		}),
	}

	client := &Client{
		config:              &Config{Model: "gpt-3.5-turbo"},
		usageCallbackConfig: config,
	}

	messages := []*schema.Message{
		{
			Role:    schema.User,
			Content: "Hello",
		},
	}

	req, _, err := client.genRequest(messages)
	if err != nil {
		t.Fatalf("Expected no error generating request, got: %v", err)
	}

	// Check if the usage parameter was added to extra fields
	extraFields := req.GetExtraFields()
	if extraFields == nil {
		t.Fatal("Expected extra fields to be set")
	}

	usageField, exists := extraFields["usage"]
	if !exists {
		t.Error("Expected 'usage' field in extra fields")
	}

	usageMap, ok := usageField.(map[string]any)
	if !ok {
		t.Error("Expected usage field to be a map")
	}

	include, exists := usageMap["include"]
	if !exists {
		t.Error("Expected 'include' field in usage map")
	}

	if include != true {
		t.Errorf("Expected include=true, got=%v", include)
	}
}

func TestIncludeUsageInRequestDisabled(t *testing.T) {
	config := &UsageCallbackConfig{
		Enabled:               true,
		IncludeUsageInRequest: false, // Disabled
		Handler: UsageCallbackFunc(func(ctx context.Context, usage *ExtendedTokenUsage) error {
			return nil
		}),
	}

	client := &Client{
		config:              &Config{Model: "gpt-3.5-turbo"},
		usageCallbackConfig: config,
	}

	messages := []*schema.Message{
		{
			Role:    schema.User,
			Content: "Hello",
		},
	}

	req, _, err := client.genRequest(messages)
	if err != nil {
		t.Fatalf("Expected no error generating request, got: %v", err)
	}

	// Check that usage parameter was NOT added
	extraFields := req.GetExtraFields()
	if extraFields != nil {
		if _, exists := extraFields["usage"]; exists {
			t.Error("Expected 'usage' field NOT to be in extra fields when disabled")
		}
	}
}
