/**
 * KERNELIZE Platform - Landing Page
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { Metadata } from 'next'
import { Inter, Plus_Jakarta_Sans, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import Navbar from '@/components/layout/Navbar'
import Footer from '@/components/layout/Footer'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ['latin'],
  variable: '--font-plus-jakarta-sans',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
})

export const metadata: Metadata = {
  title: {
    default: 'KERNELIZE Platform | Enterprise AI & Data Management Infrastructure',
    template: '%s | KERNELIZE Platform',
  },
  description: 'Comprehensive enterprise-grade platform implementing advanced AI compression, real-time analytics, data management, and DevOps operations. Deploy production-ready in minutes with 99.99% uptime SLA.',
  keywords: [
    'AI compression',
    'data pipeline',
    'enterprise analytics',
    'machine learning',
    'DevOps',
    'Kubernetes',
    'cloud integration',
    'data management',
    'real-time processing',
    'infrastructure platform',
  ],
  authors: [{ name: 'KERNELIZE Platform' }],
  openGraph: {
    title: 'KERNELIZE Platform | Enterprise AI & Data Management',
    description: 'Comprehensive enterprise-grade platform for AI compression, analytics, and data management.',
    type: 'website',
    locale: 'en_US',
    siteName: 'KERNELIZE Platform',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'KERNELIZE Platform - Enterprise AI Infrastructure',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'KERNELIZE Platform | Enterprise AI & Data Management',
    description: 'Comprehensive enterprise-grade platform for AI compression, analytics, and data management.',
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.variable} ${plusJakartaSans.variable} ${jetbrainsMono.variable} bg-background text-text-primary antialiased`}>
        <Navbar />
        <main className="min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
