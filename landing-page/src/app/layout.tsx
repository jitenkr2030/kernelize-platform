import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'KERNELIZE - Compress the World\'s Knowledge',
  description: 'Build the world\'s first Knowledge Compression Infrastructure. Compress knowledge 100×–10,000× with semantic retention.',
  keywords: ['AI', 'knowledge compression', 'semantic search', 'LLM', 'machine learning'],
  openGraph: {
    title: 'KERNELIZE - Knowledge Compression Infrastructure',
    description: 'Compress the world\'s knowledge into ultra-dense intelligence kernels.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}
