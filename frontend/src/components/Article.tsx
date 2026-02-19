import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import { 
  Box, Typography, Link,
  Container, Table, TableBody,
  TableCell, TableContainer,
  TableHead, TableRow, Paper
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
// import { materialLight } from "react-syntax-highlighter/dist/esm/styles/prism";

interface ArticleProps {
  content: string;
}

export default function Article({ content }: ArticleProps) {
  
  const theme = useTheme();

  const codeStyle = vscDarkPlus; // theme.palette.mode === "dark" ? vscDarkPlus : materialLight;

  const components: any = {
    h1: (props: any) => (
      <Typography
        variant="h4"
        component="h2"
        fontWeight="200"
        sx={{ mt: 2, mb: 3, letterSpacing: "-0.01em", textAlign: "left", color: theme.palette.text.primary }}
        {...props}
      />
    ),
    h2: (props: any) => (
      <Typography
        variant="h4"
        component="h2"
        fontWeight="200"
        sx={{ mt: 6, mb: 3, letterSpacing: "-0.01em", textAlign: "left", color: theme.palette.text.primary }}
        {...props}
      />
    ),
    h3: (props: any) => (
      <Typography
        variant="h6"
        component="h3"
        fontWeight="100"
        sx={{ mt: 4, mb: 2, textAlign: "left", color: theme.palette.text.primary }}
        {...props}
      />
    ),
    p: (props: any) => (
      <Typography
        variant="body1"
        component="div"
        sx={{
          mb: 3,
          fontSize: "1.125rem",
          lineHeight: 1.75,
          textAlign: "left",
          color: theme.palette.text.primary,
        }}
        {...props}
      />
    ),
    img: ({ alt, src }: any) => (
      <Box
        component="figure"
        sx={{
          my: 5,
          mx: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Box
          component="img"
          src={src}
          alt={alt}
          sx={{
            maxWidth: "100%",
            borderRadius: 2,
            border: `1px solid ${theme.palette.divider}`,
            maxHeight: "600px",
            objectFit: "contain",
          }}
        />

        {alt && (
          <Typography
            component="figcaption"
            variant="caption"
            sx={{
              mt: 1.5,
              fontSize: "0.9rem",
              textAlign: "center",
              fontStyle: "italic",
              maxWidth: "80%",
              color: theme.palette.text.secondary,
            }}
          >
            {alt}
          </Typography>
        )}
      </Box>
    ),
    ul: (props: any) => (
      <Box component="ul" sx={{ pl: 3, mb: 3, textAlign: "left", width: "100%", color: theme.palette.text.primary }}>
        {props.children}
      </Box>
    ),
    ol: (props: any) => (
      <Box component="ol" sx={{ pl: 3, mb: 3, textAlign: "left", width: "100%", color: theme.palette.text.primary }}>
        {props.children}
      </Box>
    ),
    li: (props: any) => (
      <li style={{ marginBottom: "8px", paddingLeft: "4px" }}>
        <Typography component="span" variant="body1" sx={{ fontSize: "1.125rem", lineHeight: 1.75, color: theme.palette.text.primary }}>
          {props.children}
        </Typography>
      </li>
    ),
    a: (props: any) => (
      <Link
        href={props.href}
        target="_blank"
        rel="noopener noreferrer"
        sx={{
          color: theme.palette.primary.main,
          textDecoration: "underline",
          textUnderlineOffset: "3px",
          fontWeight: 500,
        }}
      >
        {props.children}
      </Link>
    ),
    blockquote: (props: any) => (
      <Box
        sx={{
          borderLeft: `4px solid ${theme.palette.divider}`,
          pl: 3,
          py: 1,
          my: 4,
          ml: 0,
          mr: 0,
          fontStyle: "italic",
          color: theme.palette.text.secondary,
          background: theme.palette.mode === "dark" ? alpha(theme.palette.background.paper, 0.02) : undefined,
          textAlign: "left",
        }}
      >
        {props.children}
      </Box>
    ),
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || "");
      return !inline && match ? (
        <Box sx={{ my: 4, borderRadius: 2, overflow: "hidden", fontSize: "0.85rem", textAlign: "left" }}>
          <SyntaxHighlighter
            style={codeStyle}
            language={match[1]}
            PreTag="div"
            customStyle={{
              margin: 0,
              padding: "20px",
              color: "inherit",
            }}
            {...props}
          >
            {String(children).replace(/\n$/, "")}
          </SyntaxHighlighter>
        </Box>
      ) : (
        <Box
          component="span"
          sx={{
            bgcolor: theme.palette.mode === "dark" ? alpha(theme.palette.background.paper, 0.04) : theme.palette.background.paper,
            color: theme.palette.text.primary,
            px: 0.6,
            py: 0.2,
            borderRadius: 1,
            fontFamily: "monospace",
            fontSize: "0.9em",
            fontWeight: 500,
            border: `1px solid ${theme.palette.divider}`,
          }}
        >
          {children}
        </Box>
      );
    },
    table: (props: any) => (
      <TableContainer component={Paper} elevation={0} variant="outlined" sx={{ my: 4, borderRadius: 2, border: `1px solid ${theme.palette.divider}` }}>
        <Table {...props} />
      </TableContainer>
    ),
    thead: (props: any) => <TableHead sx={{ bgcolor: alpha(theme.palette.background.paper, 0.02) }} {...props} />,
    tbody: (props: any) => <TableBody {...props} />,
    tr: (props: any) => <TableRow sx={{ "&:last-child td, &:last-child th": { border: 0 } }} {...props} />,
    th: (props: any) => <TableCell sx={{ fontWeight: 700, color: theme.palette.text.primary, textAlign: "left", background: alpha(theme.palette.background.paper, 0.01) }} {...props} />,
    td: (props: any) => <TableCell sx={{ color: theme.palette.text.secondary, textAlign: "left" }} {...props} />,
  };

  return (
    <Container maxWidth="md" sx={{ pb: 12, marginTop: 0 }}>
      <Box component="article">
        <ReactMarkdown 
          components={components}
          remarkPlugins={[remarkGfm]}
        >
          {content}
        </ReactMarkdown>
      </Box>
    </Container>
  );
}