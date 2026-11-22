package com.jura.jura_backend.dto.rag;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Mirror of the Python {@code ExtractedFacts} model returned by the analyzer.
 * Optional on {@link RagResponseDto}; ignored cleanly when absent.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class ExtractedFactsDto {

    @JsonProperty("legal_domain")
    private String legalDomain;

    @JsonProperty("extracted_facts")
    private List<String> extractedFacts;

    @JsonProperty("missing_facts")
    private List<String> missingFacts;

    @JsonProperty("needs_clarification")
    private Boolean needsClarification;

    @JsonProperty("clarification_question")
    private String clarificationQuestion;
}
