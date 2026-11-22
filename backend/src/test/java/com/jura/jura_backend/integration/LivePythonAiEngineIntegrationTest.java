package com.jura.jura_backend.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jura.jura_backend.dto.auth.AuthResponse;
import com.jura.jura_backend.dto.auth.RegisterRequest;
import com.jura.jura_backend.dto.bookmark.BookmarkResponse;
import com.jura.jura_backend.dto.bookmark.CreateBookmarkRequest;
import com.jura.jura_backend.dto.bookmark.UpdateBookmarkRequest;
import com.jura.jura_backend.dto.chat.ChatMessageResponse;
import com.jura.jura_backend.dto.chat.ChatResponse;
import com.jura.jura_backend.dto.chat.ChatSessionResponse;
import com.jura.jura_backend.dto.chat.CreateSessionRequest;
import com.jura.jura_backend.dto.chat.SendMessageRequest;
import com.jura.jura_backend.dto.legalitem.PageResponse;
import com.jura.jura_backend.dto.rag.CitationResponse;
import com.jura.jura_backend.dto.user.UserResponse;
import com.jura.jura_backend.entity.LegalItem;
import com.jura.jura_backend.repository.LegalItemRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
// Opt-in only: this test drives a REAL running python-ai-engine (and therefore a
// real Qdrant + Gemini call). Without this guard, `mvn clean test` fails on any
// machine that has not started the AI engine, which made the README's test
// instructions misleading. Run it explicitly:
//     $env:JURA_LIVE_AI="true"; mvn test -Dtest=LivePythonAiEngineIntegrationTest
@EnabledIfEnvironmentVariable(named = "JURA_LIVE_AI", matches = "true")
class LivePythonAiEngineIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private LegalItemRepository legalItemRepository;

    private UUID seededLegalItemId;

    @BeforeEach
    void setUp() {
        seededLegalItemId = UUID.randomUUID();
        LegalItem legalItem = LegalItem.builder()
                .id(seededLegalItemId)
                .type("ACT")
                .title("Bharatiya Nyaya Sanhita, 2023 - Section 103")
                .citation(objectMapper.createArrayNode())
                .question("Punishment for murder")
                .year(2023)
                .source("India Code")
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
        legalItemRepository.save(legalItem);
    }

    @Test
    @DisplayName("End-to-End Test: Live connection to Python AI Engine and all feature workflows")
    void testCompleteWorkflowWithLivePythonAi() throws Exception {
        // 1. Verify Health check reporting Python AI engine UP
        MvcResult healthResult = mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andReturn();

        Map<?, ?> healthBody = objectMapper.readValue(healthResult.getResponse().getContentAsString(), Map.class);
        assertEquals("UP", healthBody.get("status"));
        Map<?, ?> components = (Map<?, ?>) healthBody.get("components");
        assertNotNull(components);
        assertEquals("UP", components.get("pythonAiEngine"));

        // 2. Register a new user
        String email = "advocate.live" + System.currentTimeMillis() + "@jura.law";
        RegisterRequest registerRequest = RegisterRequest.builder()
                .email(email)
                .password("Secr3tPassword!")
                .name("Senior Advocate Verma")
                .phoneNumber("98765" + (System.currentTimeMillis() % 100000))
                .build();

        MvcResult registerResult = mockMvc.perform(post("/api/auth/register")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(registerRequest)))
                .andExpect(status().isCreated())
                .andReturn();

        AuthResponse authResponse = objectMapper.readValue(
                registerResult.getResponse().getContentAsString(), AuthResponse.class);
        assertNotNull(authResponse.getToken());
        String jwtToken = "Bearer " + authResponse.getToken();

        // 3. Verify /api/users/me
        MvcResult userMeResult = mockMvc.perform(get("/api/users/me")
                        .header("Authorization", jwtToken))
                .andExpect(status().isOk())
                .andReturn();

        UserResponse userResponse = objectMapper.readValue(
                userMeResult.getResponse().getContentAsString(), UserResponse.class);
        assertEquals(email, userResponse.getEmail());
        assertEquals("Senior Advocate Verma", userResponse.getName());

        // 4. Create a chat session
        CreateSessionRequest sessionRequest = CreateSessionRequest.builder()
                .title("Homicide Research")
                .build();

        MvcResult sessionResult = mockMvc.perform(post("/api/chat/sessions")
                        .header("Authorization", jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(sessionRequest)))
                .andExpect(status().isCreated())
                .andReturn();

        ChatSessionResponse sessionResponse = objectMapper.readValue(
                sessionResult.getResponse().getContentAsString(), ChatSessionResponse.class);
        assertNotNull(sessionResponse.getId());
        UUID sessionId = sessionResponse.getId();

        // 5. Send message and connect to Python AI engine to get real RAG answer + citations
        SendMessageRequest messageRequest = SendMessageRequest.builder()
                .content("What is the punishment for murder under Indian law?")
                .limit(5)
                .legalStatus("CURRENT")
                .build();

        MvcResult chatResult = mockMvc.perform(post("/api/chat/sessions/" + sessionId + "/messages")
                        .header("Authorization", jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(messageRequest)))
                .andExpect(status().isOk())
                .andReturn();

        ChatResponse chatResponse = objectMapper.readValue(
                chatResult.getResponse().getContentAsString(), ChatResponse.class);

        assertNotNull(chatResponse.getMessage());
        assertEquals("ASSISTANT", chatResponse.getMessage().getRole());

        // ------------------------------------------------------------------
        // Real RAG structure validation.
        //
        // These assertions deliberately do NOT pin specific section numbers.
        // The old version asserted "Section 103" / "India Code" literals, which
        // is exactly what the hardcoded stub in the Python service returned --
        // so the test passed while proving nothing about retrieval or grounding.
        // What actually matters now is that the answer came from the pipeline:
        // non-blank generated text, citations built from chunk metadata, and
        // every citation complete enough to display and verify.
        // ------------------------------------------------------------------
        String answer = chatResponse.getMessage().getContent();
        assertNotNull(answer);
        assertFalse(answer.isBlank(), "RAG answer must not be blank");

        assertNotNull(chatResponse.getCitations());
        assertFalse(
                chatResponse.getCitations().isEmpty(),
                "a grounded answer must carry at least one citation built from chunk metadata"
        );

        for (CitationResponse citation : chatResponse.getCitations()) {
            assertNotNull(citation.getDocumentTitle(), "citation.documentTitle is required");
            assertFalse(citation.getDocumentTitle().isBlank());

            assertNotNull(citation.getArticleOrSection(), "citation.articleOrSection is required");
            assertTrue(
                    citation.getArticleOrSection().matches("^(Article|Section) \\S+.*"),
                    "articleOrSection must be a provision label, got: "
                            + citation.getArticleOrSection()
            );

            assertNotNull(citation.getPage(), "citation.page is required");
            assertTrue(citation.getPage() > 0, "citation.page must be a real PDF page");

            assertNotNull(citation.getSource(), "citation.source is required");
            assertFalse(citation.getSource().isBlank());
        }

        // The deterministic verifier removes any marker that does not resolve to
        // a supplied reference, so at least one verified [n] must survive in a
        // cited answer. A leftover marker with no citation would mean the
        // verifier let something ungrounded through.
        assertTrue(
                answer.matches("(?s).*\\[\\d+\\].*"),
                "answer should retain at least one verified [n] citation marker"
        );

        // 6. Verify message history retrieval for session
        MvcResult messagesListResult = mockMvc.perform(get("/api/chat/sessions/" + sessionId + "/messages")
                        .header("Authorization", jwtToken))
                .andExpect(status().isOk())
                .andReturn();

        List<?> messageList = objectMapper.readValue(messagesListResult.getResponse().getContentAsString(), List.class);
        assertEquals(2, messageList.size()); // 1 user message, 1 assistant message

        // 7. Test Legal Items API
        mockMvc.perform(get("/api/legal-items")
                        .header("Authorization", jwtToken)
                        .param("type", "ACT"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.content").isArray());

        // 8. Test Bookmarks: Create Bookmark
        CreateBookmarkRequest bookmarkRequest = CreateBookmarkRequest.builder()
                .legalItemId(seededLegalItemId)
                .note("Reference for homicide defense brief")
                .build();

        MvcResult createBookmarkResult = mockMvc.perform(post("/api/bookmarks")
                        .header("Authorization", jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(bookmarkRequest)))
                .andExpect(status().isCreated())
                .andReturn();

        BookmarkResponse bookmarkResponse = objectMapper.readValue(
                createBookmarkResult.getResponse().getContentAsString(), BookmarkResponse.class);
        assertEquals(seededLegalItemId, bookmarkResponse.getLegalItemId());

        // 9. Test Bookmarks: List Bookmarks
        mockMvc.perform(get("/api/bookmarks")
                        .header("Authorization", jwtToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].legalItemId").value(seededLegalItemId.toString()));

        // 10. Test Bookmarks: Update Bookmark Note
        UpdateBookmarkRequest updateNoteRequest = UpdateBookmarkRequest.builder()
                .note("Updated note for trial preparation")
                .build();

        mockMvc.perform(patch("/api/bookmarks/" + seededLegalItemId)
                        .header("Authorization", jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(updateNoteRequest)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.note").value("Updated note for trial preparation"));

        // 11. Test Bookmarks: Delete Bookmark
        mockMvc.perform(delete("/api/bookmarks/" + seededLegalItemId)
                        .header("Authorization", jwtToken))
                .andExpect(status().isNoContent());
    }
}
