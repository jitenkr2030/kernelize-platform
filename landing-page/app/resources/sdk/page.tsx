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
import { Zap, Code, Download, ArrowRight, Terminal, Check } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'
import { useState } from 'react'

const sdks = [
  {
    id: 'javascript',
    name: 'JavaScript / TypeScript',
    description: 'Official SDK for Node.js, browser, and edge environments.',
    version: 'v3.2.0',
    packageManager: 'npm install @kernelize/sdk',
    installCommand: 'npm install @kernelize/sdk',
    docsLink: '/docs/sdk/javascript',
    features: [
      'TypeScript support out of the box',
      'Promise-based API',
      'Browser and Node.js compatible',
      'Edge runtime support',
    ],
  },
  {
    id: 'python',
    name: 'Python',
    description: 'Official SDK for Python applications and data workflows.',
    version: 'v3.1.0',
    packageManager: 'pip install kernelize-sdk',
    installCommand: 'pip install kernelize-sdk',
    docsLink: '/docs/sdk/python',
    features: [
      'Async support with asyncio',
      'Pydantic models',
      'Type hints support',
      'Works with FastAPI, Flask, Django',
    ],
  },
  {
    id: 'go',
    name: 'Go',
    description: 'Official SDK for high-performance Go applications.',
    version: 'v2.4.0',
    packageManager: 'go get github.com/kernelize/sdk-go',
    installCommand: 'go get github.com/kernelize/sdk-go',
    docsLink: '/docs/sdk/go',
    features: [
      'Context support',
      'Strongly typed models',
      'HTTP client customization',
      'Middleware support',
    ],
  },
  {
    id: 'ruby',
    name: 'Ruby',
    description: 'Official SDK for Ruby and Ruby on Rails applications.',
    version: 'v2.3.0',
    packageManager: 'gem install kernelize-sdk',
    installCommand: 'gem install kernelize-sdk',
    docsLink: '/docs/sdk/ruby',
    features: [
      'Rails integration',
      'ActiveRecord support',
      'Rake tasks',
      'Convention over configuration',
    ],
  },
  {
    id: 'java',
    name: 'Java',
    description: 'Official SDK for enterprise Java applications.',
    version: 'v2.2.0',
    packageManager: 'mvn dependency:kernelize-sdk',
    installCommand: '<dependency>\n  <groupId>com.kernelize</groupId>\n  <artifactId>sdk</artifactId>\n  <version>2.2.0</version>\n</dependency>',
    docsLink: '/docs/sdk/java',
    features: [
      'Spring Boot starter',
      'Reactive support',
      'HTTP client configurability',
      'Annotation processing',
    ],
  },
  {
    id: 'rust',
    name: 'Rust',
    description: 'Official SDK for performance-critical Rust applications.',
    version: 'v1.5.0',
    packageManager: 'cargo add kernelize-sdk',
    installCommand: 'cargo add kernelize-sdk',
    docsLink: '/docs/sdk/rust',
    features: [
      'Async/await support',
      'Zero-copy serialization',
      'Strong type safety',
      'WebAssembly compatible',
    ],
  },
]

export default function SdkDownloadsPage() {
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null)

  const copyCommand = (command: string) => {
    navigator.clipboard.writeText(command)
    setCopiedCommand(command)
    setTimeout(() => setCopiedCommand(null), 2000)
  }

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
              SDK Downloads
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Official software development kits for your favorite languages and frameworks.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Quick Install */}
      <section className="py-16">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Quick Install
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Get started with the KERNELIZE SDK in your language of choice.
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto space-y-6">
            {sdks.map((sdk, index) => (
              <motion.div
                key={sdk.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" className="p-6" id={sdk.id}>
                  <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-6">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <h3 className="text-xl font-semibold text-text-primary">
                          {sdk.name}
                        </h3>
                        <span className="px-2 py-1 rounded text-xs font-medium bg-primary/10 text-primary">
                          {sdk.version}
                        </span>
                      </div>
                      <p className="text-text-secondary mb-4">
                        {sdk.description}
                      </p>
                      <ul className="space-y-2">
                        {sdk.features.map((feature) => (
                          <li key={feature} className="flex items-center gap-2 text-sm text-text-secondary">
                            <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="flex-shrink-0 lg:w-80">
                      <div className="bg-slate-900 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-text-secondary text-sm">Install Command</span>
                          <button
                            onClick={() => copyCommand(sdk.installCommand)}
                            className="text-text-secondary hover:text-text-primary transition-colors"
                          >
                            {copiedCommand === sdk.installCommand ? (
                              <Check className="w-4 h-4 text-green-500" />
                            ) : (
                              <Terminal className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                        <code className="text-sm text-slate-300 block overflow-x-auto">
                          {sdk.installCommand}
                        </code>
                      </div>
                      <div className="mt-4 flex flex-col gap-2">
                        <Link href={sdk.docsLink}>
                          <Button variant="secondary" size="sm" className="w-full">
                            View Documentation
                            <ArrowRight className="w-4 h-4 ml-2" />
                          </Button>
                        </Link>
                        <a
                          href={`https://github.com/kernelize/sdk-${sdk.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <Button variant="ghost" size="sm" className="w-full">
                            View on GitHub
                          </Button>
                        </a>
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Community SDKs */}
      <section className="py-16">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Community SDKs
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Additional SDKs maintained by our community.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {[
              { name: '.NET/C#', maintainer: 'Community', repo: '#' },
              { name: 'PHP', maintainer: 'Community', repo: '#' },
              { name: 'Elixir', maintainer: 'Community', repo: '#' },
            ].map((sdk, index) => (
              <motion.div
                key={sdk.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" hover className="p-6 h-full">
                  <h3 className="font-semibold text-text-primary mb-2">
                    {sdk.name}
                  </h3>
                  <p className="text-text-secondary text-sm mb-4">
                    Maintained by {sdk.maintainer}
                  </p>
                  <a
                    href={sdk.repo}
                    className="text-primary hover:text-primary-hover text-sm font-medium inline-flex items-center gap-1"
                  >
                    View Repository
                    <ArrowRight className="w-4 h-4" />
                  </a>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Getting Started */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Need Help Getting Started?
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                Check out our documentation or reach out to our community for assistance.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/docs">
                  <Button size="lg">
                    Read the Docs
                  </Button>
                </Link>
                <Link href="/community">
                  <Button variant="secondary" size="lg">
                    Join Community
                  </Button>
                </Link>
              </div>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
