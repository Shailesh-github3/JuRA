package com.jura.jura_backend.client;

import com.jura.jura_backend.dto.rag.RagQueryRequest;
import com.jura.jura_backend.exception.PythonServiceException;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that {@link PythonRagClient} keeps the status code and the engine's
 * own detail message when the Python side reports an error. This matters most
 * for 503: the engine deliberately returns 503 with a message like
 * "GOOGLE_API_KEY is not configured", and that must reach the API client as a
 * 503 with the same explanation instead of being masked as a generic 502.
 *
 * Uses the JDK's built-in {@link HttpServer} so no extra test dependencies
 * or mocks are required.
 */
class PythonRagClientTest {

    private HttpServer server;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    private PythonRagClient clientReturning(int status, String body) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/query", exchange -> {
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        server.start();
        return new PythonRagClient(
                "http://127.0.0.1:" + server.getAddress().getPort(),
                2000,
                5000
        );
    }

    private RagQueryRequest anyRequest() {
        return RagQueryRequest.builder().query("punishment for murder").limit(3).build();
    }

    @Test
    @DisplayName("503 from the engine is propagated as 503 with the engine's detail message")
    void propagates503WithEngineDetail() throws Exception {
        PythonRagClient client = clientReturning(
                503,
                "{\"detail\":\"GOOGLE_API_KEY is not configured on the AI engine\"}"
        );

        PythonServiceException ex = assertThrows(
                PythonServiceException.class,
                () -> client.query(anyRequest())
        );

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.getStatus());
        assertTrue(
                ex.getMessage().contains("GOOGLE_API_KEY"),
                "the engine's detail message must survive, got: " + ex.getMessage()
        );
    }

    @Test
    @DisplayName("503 body that is not JSON still propagates as 503")
    void propagates503WithNonJsonBody() throws Exception {
        PythonRagClient client = clientReturning(503, "upstream timeout");

        PythonServiceException ex = assertThrows(
                PythonServiceException.class,
                () -> client.query(anyRequest())
        );

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.getStatus());
        assertTrue(ex.getMessage().contains("upstream timeout"));
    }

    @Test
    @DisplayName("other 5xx errors still map to 502 Bad Gateway")
    void mapsOther5xxTo502() throws Exception {
        PythonRagClient client = clientReturning(500, "{\"detail\":\"boom\"}");

        PythonServiceException ex = assertThrows(
                PythonServiceException.class,
                () -> client.query(anyRequest())
        );

        assertEquals(HttpStatus.BAD_GATEWAY, ex.getStatus());
    }

    @Test
    @DisplayName("4xx errors still map to 400 Bad Request")
    void maps4xxTo400() throws Exception {
        PythonRagClient client = clientReturning(422, "{\"detail\":\"invalid limit\"}");

        PythonServiceException ex = assertThrows(
                PythonServiceException.class,
                () -> client.query(anyRequest())
        );

        assertEquals(HttpStatus.BAD_REQUEST, ex.getStatus());
    }
}
