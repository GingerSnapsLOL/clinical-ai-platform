import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Clinical AI Platform",
  description: "Frontend for the Clinical AI Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
