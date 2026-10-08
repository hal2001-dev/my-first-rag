

package com.example.document.util;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import javax.activation.DataHandler;
import javax.mail.Part;
import javax.mail.Session;
import javax.mail.internet.*;
import javax.mail.util.ByteArrayDataSource;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Attribute;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

public class HtmlToMhtConverter {

    private static final int MAX_IMAGE_BYTES =
            10 * 1024 * 1024;

    private static final long MAX_TOTAL_BYTES =
            50L * 1024 * 1024;

    private static final int MAX_IMAGES = 100;

    private static final int CONNECT_TIMEOUT = 5000;
    private static final int READ_TIMEOUT = 15000;

    private final Set<String> allowedHosts;

    public HtmlToMhtConverter(Set<String> allowedHosts) {
        if (allowedHosts == null) {
            throw new IllegalArgumentException(
                    "allowedHosts is null");
        }

        this.allowedHosts = new HashSet<>();

        for (String host : allowedHosts) {
            this.allowedHosts.add(
                    host.toLowerCase(Locale.ROOT));
        }
    }

    /**
     * HTML 문자열을 MHT 파일로 변환합니다.
     */
    public void convert(
            String html,
            String baseUrl,
            Path outputFile) throws Exception {

        try (OutputStream out =
                     new BufferedOutputStream(
                             Files.newOutputStream(outputFile))) {

            convert(html, baseUrl, out);
        }
    }

    /**
     * OutputStream으로 MHT를 생성합니다.
     */
    public void convert(
            String html,
            String baseUrl,
            OutputStream out) throws Exception {

        if (html == null || out == null) {
            throw new IllegalArgumentException(
                    "HTML or output is null");
        }

        // 1. HTML 파싱
        Document document = Jsoup.parse(
                html,
                baseUrl == null ? "" : baseUrl
        );

        // 2. 불필요한 태그와 속성 제거
        normalizeHtml(document);

        // 3. 이미지 수집 및 CID 치환
        Map<String, ImageResource> images =
                collectImages(document);

        // 4. MIME 생성
        MimeMultipart multipart =
                new MimeMultipart("related");

        String rootCid = createCid();

        MimeBodyPart htmlPart = new MimeBodyPart();

        htmlPart.setText(
                document.outerHtml(),
                "UTF-8",
                "html"
        );

        htmlPart.setContentID("<" + rootCid + ">");

        multipart.addBodyPart(htmlPart);

        // 5. 이미지 MIME 파트 생성
        for (ImageResource image : images.values()) {

            MimeBodyPart imagePart =
                    new MimeBodyPart();

            ByteArrayDataSource dataSource =
                    new ByteArrayDataSource(
                            image.data,
                            image.contentType
                    );

            imagePart.setDataHandler(
                    new DataHandler(dataSource)
            );

            imagePart.setContentID(
                    "<" + image.cid + ">"
            );

            imagePart.setDisposition(Part.INLINE);

            multipart.addBodyPart(imagePart);
        }

        // 6. 최종 MHT 메시지 생성
        Session session = Session.getInstance(
                new Properties()
        );

        MimeMessage message =
                new MimeMessage(session);

        message.setSubject(
                "HTML Document", "UTF-8");

        message.setContent(multipart);
        message.saveChanges();

        // multipart/related의 루트 HTML 지정
        ContentType type =
                new ContentType(message.getContentType());

        type.setParameter("type", "text/html");
        type.setParameter(
                "start", "<" + rootCid + ">");

        message.setHeader(
                "Content-Type", type.toString());

        message.writeTo(out);
        out.flush();
    }

    /**
     * HTML의 의미 있는 문서 구조를 유지합니다.
     */
    private void normalizeHtml(Document document) {

        document.select(
                "script, style, link, iframe, " +
                "object, embed, canvas, " +
                "video, audio, form, button, " +
                "input, select, textarea, " +
                "source"
        ).remove();

        for (Element element :
                document.getAllElements()) {

            element.removeAttr("style");
            element.removeAttr("class");
            element.removeAttr("contenteditable");

            // 이벤트 핸들러 속성 제거
            List<Attribute> attrs =
                    new ArrayList<>(
                            element.attributes().asList());

            for (Attribute attr : attrs) {
                if (attr.getKey()
                        .toLowerCase(Locale.ROOT)
                        .startsWith("on")) {

                    element.removeAttr(attr.getKey());
                }
            }
        }

        document.outputSettings()
                .charset(StandardCharsets.UTF_8)
                .prettyPrint(false);
    }

    /**
     * HTML의 이미지를 수집하고
     * src 속성을 cid: 형식으로 변환합니다.
     */
    private Map<String, ImageResource> collectImages(
            Document document) throws IOException {

        Map<String, ImageResource> resources =
                new LinkedHashMap<>();

        long totalBytes = 0;

        for (Element img : document.select("img")) {

            String src = img.attr("src").trim();

            // lazy loading 이미지 지원
            if (src.isEmpty()) {
                src = img.attr("data-src").trim();
            }

            if (src.isEmpty()) {
                throw new IOException(
                        "Image src is empty");
            }

            // data-src 상대경로도 해석되도록 설정
            img.attr("src", src);

            String key;
            ImageResource resource;

            if (src.toLowerCase(Locale.ROOT)
                    .startsWith("data:")) {

                key = src;

                resource = resources.get(key);

                if (resource == null) {
                    resource = decodeDataImage(src);
                }

            } else {

                // 절대 및 상대 URL 처리
                key = img.absUrl("src");

                if (key.isEmpty()) {
                    throw new IOException(
                            "Invalid image URL: " + src);
                }

                resource = resources.get(key);

                if (resource == null) {
                    resource = downloadImage(key);
                }
            }

            if (!resources.containsKey(key)) {

                if (resources.size() >= MAX_IMAGES) {
                    throw new IOException(
                            "Too many images");
                }

                totalBytes += resource.data.length;

                if (totalBytes > MAX_TOTAL_BYTES) {
                    throw new IOException(
                            "Total image size exceeded");
                }

                resources.put(key, resource);
            }

            // 원본 URL 대신 내부 CID 사용
            img.attr("src", "cid:" + resource.cid);

            img.removeAttr("srcset");
            img.removeAttr("data-src");
            img.removeAttr("data-srcset");
        }

        return resources;
    }

    /**
     * HTTP/HTTPS 이미지를 다운로드합니다.
     */
    private ImageResource downloadImage(
            String imageUrl) throws IOException {

        URL url = new URL(imageUrl);

        String protocol =
                url.getProtocol().toLowerCase(Locale.ROOT);

        if (!"http".equals(protocol)
                && !"https".equals(protocol)) {

            throw new IOException(
                    "Unsupported protocol: " + protocol);
        }

        String host =
                url.getHost().toLowerCase(Locale.ROOT);

        // 서버가 접근해도 되는 이미지 호스트만 허용
        if (!allowedHosts.contains(host)
                || url.getUserInfo() != null) {

            throw new IOException(
                    "Image host not allowed: " + host);
        }

        HttpURLConnection connection =
                (HttpURLConnection) url.openConnection();

        connection.setConnectTimeout(CONNECT_TIMEOUT);
        connection.setReadTimeout(READ_TIMEOUT);

        // 다른 호스트로 우회하는 리다이렉트 차단
        connection.setInstanceFollowRedirects(false);

        connection.setRequestMethod("GET");
        connection.setRequestProperty(
                "Accept", "image/*");

        try {
            int status = connection.getResponseCode();

            if (status != HttpURLConnection.HTTP_OK) {
                throw new IOException(
                        "Image HTTP status: " + status
                        + ", URL: " + imageUrl);
            }

            String contentType = normalizeContentType(
                    connection.getContentType());

            validateImageType(contentType);

            if (connection.getContentLengthLong()
                    > MAX_IMAGE_BYTES) {

                throw new IOException(
                        "Image size exceeded");
            }

            try (InputStream in =
                         connection.getInputStream()) {

                byte[] bytes = readLimited(in);

                return new ImageResource(
                        createCid(),
                        bytes,
                        contentType
                );
            }

        } finally {
            connection.disconnect();
        }
    }

    /**
     * data:image/...;base64 형식 처리
     */
    private ImageResource decodeDataImage(
            String src) throws IOException {

        int comma = src.indexOf(',');

        if (comma < 0) {
            throw new IOException(
                    "Invalid Base64 image");
        }

        String header = src.substring(5, comma)
                .toLowerCase(Locale.ROOT);

        if (!header.endsWith(";base64")) {
            throw new IOException(
                    "Only Base64 data images supported");
        }

        String contentType =
                normalizeContentType(
                        header.substring(
                                0, header.length() - 7));

        validateImageType(contentType);

        String encoded = src.substring(comma + 1)
                .replaceAll("\\s+", "");

        if (encoded.length()
                > ((long) MAX_IMAGE_BYTES + 2) / 3 * 4 + 4) {

            throw new IOException(
                    "Base64 image size exceeded");
        }

        try {
            byte[] bytes =
                    Base64.getDecoder().decode(encoded);

            if (bytes.length == 0
                    || bytes.length > MAX_IMAGE_BYTES) {

                throw new IOException(
                        "Invalid image size");
            }

            return new ImageResource(
                    createCid(),
                    bytes,
                    contentType
            );

        } catch (IllegalArgumentException e) {
            throw new IOException(
                    "Invalid Base64 data", e);
        }
    }

    private byte[] readLimited(InputStream in)
            throws IOException {

        ByteArrayOutputStream out =
                new ByteArrayOutputStream();

        byte[] buffer = new byte[8192];
        int len;

        while ((len = in.read(buffer)) != -1) {

            if (out.size() + len > MAX_IMAGE_BYTES) {
                throw new IOException(
                        "Image size exceeded");
            }

            out.write(buffer, 0, len);
        }

        if (out.size() == 0) {
            throw new IOException("Empty image");
        }

        return out.toByteArray();
    }

    private String normalizeContentType(String value) {

        if (value == null) {
            return "";
        }

        String type = value.split(";", 2)[0]
                .trim()
                .toLowerCase(Locale.ROOT);

        if ("image/jpg".equals(type)) {
            return "image/jpeg";
        }

        return type;
    }

    private void validateImageType(String type)
            throws IOException {

        if (!Arrays.asList(
                "image/png",
                "image/jpeg",
                "image/gif",
                "image/webp",
                "image/bmp"
        ).contains(type)) {

            throw new IOException(
                    "Unsupported image type: " + type);
        }
    }

    private String createCid() {
        return UUID.randomUUID().toString()
                + "@mht.local";
    }

    private static class ImageResource {

        final String cid;
        final byte[] data;
        final String contentType;

        ImageResource(
                String cid,
                byte[] data,
                String contentType) {

            this.cid = cid;
            this.data = data;
            this.contentType = contentType;
        }
    }
}
