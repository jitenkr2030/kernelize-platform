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
import { Inter, Plus_Jakarta_Sans } from 'next/font/google'
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

export const metadata: Metadata = {
  title: 'KERNELIZE Platform | Enterprise AI & Data Management',
  description: 'Comprehensive enterprise-grade platform implementing advanced AI compression, analytics, data management, and DevOps operations. Deploy production-ready with complete infrastructure.',
  keywords: [
    'AI compression',
    'data pipeline',
    'enterprise analytics',
    'machine learning',
    'DevOps',
    'Kubernetes',
    'cloud integration',
    'data management',
  ],
  authors: [{ name: 'KERNELIZE Platform' }],
  openGraph: {
    title: 'KERNELIZE Platform | Enterprise AI & Data Management',
    description: 'Comprehensive enterprise-grade platform for AI compression, analytics, and data management.',
    type: 'website',
    locale: 'en_US',
    siteName: 'KERNELIZE Platform',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'KERNELIZE Platform | Enterprise AI & Data Management',
    description: 'Comprehensive enterprise-grade platform for AI compression, analytics, and data management.',
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.variable} ${plusJakartaSans.variable} bg-background text-text-primary antialiased`}>
        <Navbar />
        <main className="min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
