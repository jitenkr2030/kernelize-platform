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
import { Zap, Code, Book, Terminal, ArrowRight, Copy, Check } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'
import { useState } from 'react'

const endpoints = [
  {
    method: 'GET',
    path: '/api/v1/projects',
    description: 'List all projects',
  },
  {
    method: 'POST',
    path: '/api/v1/projects',
    description: 'Create a new project',
  },
  {
    method: 'GET',
    path: '/api/v1/deployments',
    description: 'List all deployments',
  },
  {
    method: 'POST',
    path: '/api/v1/deployments',
    description: 'Create a new deployment',
  },
  {
    method: 'GET',
    path: '/api/v1/metrics',
    description: 'Get deployment metrics',
  },
  {
    method: 'GET',
    path: '/api/v1/logs',
    description: 'Get deployment logs',
  },
]

const sdks = [
  { name: 'JavaScript', icon: 'js', description: 'For Node.js and browser applications' },
  { name: 'Python', icon: 'py', description: 'For data science and backend applications' },
  { name: 'Go', icon: 'go', description: 'For high-performance systems' },
  { name: 'Ruby', icon: 'rb', description: 'For Rails and Sinatra applications' },
  { name: 'Java', icon: 'java', description: 'For enterprise applications' },
  { name: 'Rust', icon: 'rs', description: 'For performance-critical applications' },
]

export default function ApiReferencePage() {
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null)

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedEndpoint(text)
    setTimeout(() => setCopiedEndpoint(null), 2000)
  }

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET': return 'bg-blue-500'
      case 'POST': return 'bg-green-500'
      case 'PUT': return 'bg-yellow-500'
      case 'DELETE': return 'bg-red-500'
      default: return 'bg-slate-500'
    }
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
              API Reference
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Complete documentation for the KERNELIZE REST API. Build powerful integrations with our platform.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/docs">
                <Button size="lg">
                  Read the Docs
                  <Book className="w-4 h-4 ml-2" />
                </Button>
              </Link>
              <a
                href="https://api.kernelize.com/docs"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="secondary" size="lg">
                  OpenAPI Spec
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Quick Start */}
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
              Quick Start
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Get started with the KERNELIZE API in minutes.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
            <Card variant="bordered" className="p-6">
              <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                <Terminal className="w-5 h-5 text-primary" />
                cURL Example
              </h3>
              <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm text-slate-300">
{`curl -X GET \\
  https://api.kernelize.com/v1/projects \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"`}
                </pre>
              </div>
            </Card>

            <Card variant="bordered" className="p-6">
              <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                <Code className="w-5 h-5 text-primary" />
                JavaScript Example
              </h3>
              <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm text-slate-300">
{`const response = await fetch(
  'https://api.kernelize.com/v1/projects',
  {
    headers: {
      'Authorization': 'Bearer YOUR_API_KEY',
      'Content-Type': 'application/json'
    }
  }
);`}
                </pre>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Common Endpoints */}
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
              Common Endpoints
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Explore the most commonly used API endpoints.
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto">
            <div className="space-y-4">
              {endpoints.map((endpoint, index) => (
                <motion.div
                  key={endpoint.path}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <Card variant="bordered" className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <span className={`px-3 py-1 rounded text-xs font-bold text-white ${getMethodColor(endpoint.method)}`}>
                          {endpoint.method}
                        </span>
                        <code className="text-sm text-text-primary font-mono">
                          {endpoint.path}
                        </code>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-text-secondary text-sm hidden sm:block">
                          {endpoint.description}
                        </span>
                        <button
                          onClick={() => copyToClipboard(endpoint.path)}
                          className="p-2 text-text-secondary hover:text-text-primary transition-colors"
                        >
                          {copiedEndpoint === endpoint.path ? (
                            <Check className="w-4 h-4 text-green-500" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* SDKs */}
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
              Official SDKs
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Use our official SDKs for a better development experience.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {sdks.map((sdk, index) => (
              <motion.div
                key={sdk.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Link href={`/resources/sdk#${sdk.name.toLowerCase()}`}>
                  <Card variant="bordered" hover className="p-6 h-full">
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <Code className="w-6 h-6 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-text-primary mb-1">
                          {sdk.name}
                        </h3>
                        <p className="text-text-secondary text-sm">
                          {sdk.description}
                        </p>
                      </div>
                    </div>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Authentication */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Authentication
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                All API requests require authentication using an API key. You can generate and manage your API keys from the dashboard.
              </p>
              <Link href="/signup">
                <Button size="lg">
                  Get Your API Key
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </section>

      {/* Rate Limits */}
      <section className="py-16">
        <div className="container-custom">
          <div className="max-w-4xl mx-auto text-center">
            <p className="text-text-secondary">
              For detailed information about rate limits, error codes, and advanced usage,{' '}
              <Link href="/docs" className="text-primary hover:text-primary-hover transition-colors">
                check out the full documentation
              </Link>
              .
            </p>
          </div>
        </div>
      </section>
    </main>
  )
}
