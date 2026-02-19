import Box from "@mui/material/Box";
import { useTheme } from "@mui/material/styles";

interface FillerCardProps {
  content: any;
}

export default function FillerCard({ content }: FillerCardProps) {
  const theme = useTheme();

  return (
    <Box
      sx={{
        width: "256px",
        height: "256px",
        borderRadius: 3,
        border: `2px dashed ${theme.palette.divider}`,
        backgroundColor: theme.palette.background.paper,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        color: theme.palette.text.secondary,
        gap: 1.5,
        userSelect: "none",
      }}
    >
      {content}
    </Box>
  );
}