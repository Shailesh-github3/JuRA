package com.jura.jura_backend.dto.rag;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * One turn of a multi-turn conversation, forwarded to the Python AI engine
 * in {@link RagQueryRequest#history} so the clarification loop (Feature 4)
 * can see the prior context.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatTurnDto {

    @JsonProperty("role")
    private String role;

    @JsonProperty("content")
    private String content;
}
