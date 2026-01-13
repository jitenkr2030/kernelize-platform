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

'use client'

import { motion } from 'framer-motion'
import { Search, Book, Code, Terminal, Shield, Globe, Database, Cloud, ChevronRight, Zap } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'
import { useState } from 'react'

const docSections = [
  {
    title: 'Getting Started',
    icon: Book,
    description: 'Learn the basics and set up your first project',
    articles: [
      'Quickstart Guide',
      'Installation',
      'Your First Project',
      'Configuration',
    ],
  },
  {
    title: 'Core Concepts',
    icon: Globe,
    description: 'Understand the fundamental concepts behind KERNELIZE',
    articles: [
      'Architecture Overview',
      'Security Model',
      'Scaling & Performance',
      'High Availability',
    ],
  },
  {
    title: 'API Reference',
    icon: Code,
    description: 'Complete API documentation for developers',
    articles: [
      'REST API',
      'GraphQL API',
      'Webhooks',
      'SDK Reference',
    ],
  },
  {
    title: 'CLI & Tools',
    icon: Terminal,
    description: 'Command-line tools and development utilities',
    articles: [
      'CLI Installation',
      'Commands Reference',
      'Configuration File',
      'Environment Variables',
    ],
  },
  {
    title: 'Security',
    icon: Shield,
    description: 'Security best practices and compliance information',
    articles: [
      'Authentication',
      'Authorization',
      'Data Encryption',
      'Compliance',
    ],
  },
  {
    title: 'Integrations',
    icon: Cloud,
    description: 'Connect KERNELIZE with your existing tools',
    articles: [
      'Cloud Providers',
      'CI/CD Pipelines',
      'Monitoring',
      'Third-party Apps',
    ],
  },
]

const quickLinks = [
  { title: 'SDK Downloads', href: '/resources/sdk', icon: Database },
  { title: 'API Reference', href: '/resources/api', icon: Code },
  { title: 'Community', href: '/community', icon: Globe },
  { title: 'Support', href: '/support', icon: Zap },
]

export default function DocsPage() {
  const [searchQuery, setSearchQuery] = useState('')

  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="pt-32 pb-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />
        <div className="container-custom relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center max-w-4xl mx-auto"
          >
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-text-primary mb-6">
              Documentation
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Everything you need to know to build, deploy, and scale with KERNELIZE Platform.
            </p>

            {/* Search */}
            <div className="max-w-2xl mx-auto relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search documentation..."
                className="w-full pl-12 pr-4 py-4 bg-surface border border-slate-700 rounded-xl text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
              />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="py-8">
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
            {quickLinks.map((link, index) => (
              <motion.div
                key={link.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Link href={link.href}>
                  <Card variant="bordered" hover className="p-4 h-full text-center group">
                    <link.icon className="w-8 h-8 text-primary mx-auto mb-3 group-hover:scale-110 transition-transform" />
                    <span className="font-medium text-text-primary">{link.title}</span>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Documentation Sections */}
      <section className="py-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {docSections.map((section, index) => (
              <motion.div
                key={section.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" hover className="p-6 h-full">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <section.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold text-text-primary mb-2">
                    {section.title}
                  </h3>
                  <p className="text-text-secondary text-sm mb-4">
                    {section.description}
                  </p>
                  <ul className="space-y-2">
                    {section.articles.map((article) => (
                      <li key={article}>
                        <Link
                          href={`/docs/${article.toLowerCase().replace(/\s+/g, '-')}`}
                          className="flex items-center gap-2 text-sm text-text-secondary hover:text-primary transition-colors"
                        >
                          <ChevronRight className="w-4 h-4" />
                          {article}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Resources Section */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
              <div>
                <h2 className="text-2xl md:text-3xl font-bold text-text-primary mb-4">
                  Can&apos;t find what you&apos;re looking for?
                </h2>
                <p className="text-text-secondary">
                  Check out our community forums or contact our support team for assistance.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link href="/community">
                  <Button variant="secondary" size="lg">
                    Community Forums
                  </Button>
                </Link>
                <Link href="/support">
                  <Button size="lg">
                    Get Support
                  </Button>
                </Link>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {/* Changelog Link */}
      <section className="py-16">
        <div className="container-custom">
          <div className="text-center">
            <Link
              href="/changelog"
              className="inline-flex items-center gap-2 text-primary hover:text-primary-hover transition-colors"
            >
              <span className="text-sm font-medium">View recent changes</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>
    </main>
  )
}
