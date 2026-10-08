import java.nio.file.*;
import java.util.*;
import javax.xml.bind.DatatypeConverter;

Path mhtFile = documentMhtService.createMht(docId);

try {
    byte[] mhtBytes = Files.readAllBytes(mhtFile);

    String base64Content =
        DatatypeConverter.printBase64Binary(mhtBytes);

    Map<String, Object> payload =
        new HashMap<String, Object>();

    payload.put("documentId", docId);
    payload.put("fileName", "document.mht");
    payload.put("encoding", "base64");
    payload.put("content", base64Content);

    // Jackson 1.x 이용 가능
    org.codehaus.jackson.map.ObjectMapper mapper =
        new org.codehaus.jackson.map.ObjectMapper();

    String json = mapper.writeValueAsString(payload);

    // REST API로 JSON 전송

} finally {
    Files.deleteIfExists(mhtFile);
}
