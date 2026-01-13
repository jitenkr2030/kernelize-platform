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
import { Zap, ArrowUpRight, Shield, Globe, Database, Cloud, Tag, Calendar, ChevronRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const changelogs = [
  {
    version: '2.4.0',
    date: 'January 10, 2026',
    type: 'major',
    changes: [
      { type: 'feature', description: 'AI-Powered Compression 2.0 - Reduce storage costs by up to 70%' },
      { type: 'feature', description: 'Multi-cloud disaster recovery with RPO < 1 second' },
      { type: 'improvement', description: '50% faster build times for large applications' },
      { type: 'improvement', description: 'New dashboard with real-time metrics and custom widgets' },
    ],
  },
  {
    version: '2.3.2',
    date: 'December 28, 2025',
    type: 'patch',
    changes: [
      { type: 'fix', description: 'Fixed issue with API rate limiting during peak traffic' },
      { type: 'fix', description: 'Resolved memory leak in long-running serverless functions' },
      { type: 'improvement', description: 'Updated TLS certificates to latest standards' },
    ],
  },
  {
    version: '2.3.1',
    date: 'December 15, 2025',
    type: 'patch',
    changes: [
      { type: 'fix', description: 'Fixed authentication redirect loop for SSO users' },
      { type: 'improvement', description: 'Improved error messages for invalid configurations' },
    ],
  },
  {
    version: '2.3.0',
    date: 'December 1, 2025',
    type: 'minor',
    changes: [
      { type: 'feature', description: 'New GraphQL API with full schema introspection' },
      { type: 'feature', description: 'Custom domain support for API endpoints' },
      { type: 'improvement', description: 'Enhanced monitoring with custom alerts' },
      { type: 'deprecation', description: 'Legacy REST v1 API marked for removal in v3.0' },
    ],
  },
  {
    version: '2.2.0',
    date: 'November 15, 2025',
    type: 'minor',
    changes: [
      { type: 'feature', description: 'Team collaboration features with role-based access control' },
      { type: 'feature', description: 'Branch preview environments for pull requests' },
      { type: 'improvement', description: '20% reduction in cold start times' },
    ],
  },
  {
    version: '2.1.0',
    date: 'November 1, 2025',
    type: 'minor',
    changes: [
      { type: 'feature', description: 'Edge caching with automatic cache invalidation' },
      { type: 'feature', description: 'Built-in rate limiting and throttling' },
      { type: 'improvement', description: 'Better error handling for network failures' },
    ],
  },
  {
    version: '2.0.0',
    date: 'October 15, 2025',
    type: 'major',
    changes: [
      { type: 'feature', description: 'Complete redesign of dashboard and admin panel' },
      { type: 'feature', description: 'New CLI with improved ergonomics and autocomplete' },
      { type: 'feature', description: 'Native support for WebAssembly modules' },
      { type: 'breaking', description: 'Breaking: Updated SDK requires migration guide' },
    ],
  },
]

const changeTypeLabels = {
  feature: { label: 'New', color: 'bg-green-500' },
  improvement: { label: 'Improved', color: 'bg-blue-500' },
  fix: { label: 'Fixed', color: 'bg-yellow-500' },
  deprecation: { label: 'Deprecated', color: 'bg-orange-500' },
  breaking: { label: 'Breaking', color: 'bg-red-500' },
}

export default function ChangelogPage() {
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
              Changelog
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Stay up to date with the latest features, improvements, and fixes.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link href="/docs">
                <Button variant="secondary" size="lg">
                  View Documentation
                </Button>
              </Link>
              <a
                href="https://github.com/kernelize/platform/releases"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button size="lg">
                  GitHub Releases
                  <ArrowUpRight className="w-4 h-4 ml-2" />
                </Button>
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Changelog List */}
      <section className="py-16">
        <div className="container-custom">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              {/* Timeline Line */}
              <div className="absolute left-8 top-0 bottom-0 w-px bg-slate-700 md:left-1/2" />

              {changelogs.map((entry, index) => (
                <motion.div
                  key={entry.version}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className={`relative mb-8 ${
                    index % 2 === 0 ? 'md:flex' : 'md:flex md:flex-row-reverse'
                  }`}
                >
                  {/* Timeline Dot */}
                  <div className="absolute left-8 -translate-x-1/2 w-4 h-4 rounded-full bg-primary border-4 border-background md:left-1/2 z-10" />

                  {/* Content Card */}
                  <div className={`ml-16 md:ml-0 md:w-[calc(50%-2rem)] ${index % 2 === 0 ? 'md:mr-auto' : 'md:ml-auto'}`}>
                    <Card variant="bordered" className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium text-white ${
                          entry.type === 'major' ? 'bg-red-500' :
                          entry.type === 'minor' ? 'bg-blue-500' :
                          'bg-slate-500'
                        }`}>
                          {entry.type.toUpperCase()}
                        </span>
                        <span className="text-sm text-text-secondary">
                          v{entry.version}
                        </span>
                        <span className="text-sm text-text-secondary flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          {entry.date}
                        </span>
                      </div>

                      <ul className="space-y-3">
                        {entry.changes.map((change, changeIndex) => (
                          <li key={changeIndex} className="flex items-start gap-3">
                            <span className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${changeTypeLabels[change.type as keyof typeof changeTypeLabels].color}`} />
                            <div>
                              <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium text-white mr-2 ${changeTypeLabels[change.type as keyof typeof changeTypeLabels].color}`}>
                                {changeTypeLabels[change.type as keyof typeof changeTypeLabels].label}
                              </span>
                              <span className="text-text-secondary">
                                {change.description}
                              </span>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </Card>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Subscribe Section */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Stay Updated
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                Get notified about new releases and updates directly in your inbox.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 max-w-md mx-auto">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                />
                <Button>
                  Subscribe
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
