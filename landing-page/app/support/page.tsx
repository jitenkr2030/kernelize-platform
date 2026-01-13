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
import { Zap, Search, Book, MessageCircle, Mail, Phone, ArrowRight, ExternalLink } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'
import { useState } from 'react'

const supportResources = [
  {
    icon: Book,
    title: 'Documentation',
    description: 'Browse our comprehensive guides and API reference.',
    link: '/docs',
    color: 'from-blue-500 to-cyan-600',
  },
  {
    icon: MessageCircle,
    title: 'Community Forums',
    description: 'Get help from our community of developers.',
    link: '/community',
    color: 'from-purple-500 to-violet-600',
  },
  {
    icon: Search,
    title: 'Knowledge Base',
    description: 'Search our database of frequently asked questions.',
    link: '#',
    color: 'from-green-500 to-emerald-600',
  },
  {
    icon: Mail,
    title: 'Email Support',
    description: 'Reach our support team directly via email.',
    link: 'mailto:support@kernelize.com',
    color: 'from-orange-500 to-amber-600',
  },
]

const faqs = [
  {
    question: 'How do I reset my API key?',
    answer: 'You can reset your API key from the dashboard by navigating to Settings → API Keys → Reset Key. Note that this will immediately invalidate your current key.',
  },
  {
    question: 'What is the uptime SLA for the platform?',
    answer: 'We offer a 99.99% uptime SLA for all paid plans. See our status page for real-time uptime information and incident history.',
  },
  {
    question: 'How do I export my data?',
    answer: 'You can export all your data at any time from Settings → Data Export. We provide exports in JSON and CSV formats.',
  },
  {
    question: 'Can I downgrade my plan?',
    answer: 'Yes, you can downgrade your plan at any time. Changes will be applied at the start of your next billing cycle.',
  },
  {
    question: 'How do I delete my account?',
    answer: 'To delete your account, go to Settings → Account → Delete Account. This action is irreversible and will delete all your data within 30 days.',
  },
]

const ticketCategories = [
  'Account & Billing',
  'Technical Support',
  'Feature Request',
  'Security Issue',
  'Sales Inquiry',
  'Other',
]

export default function SupportPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [openFaq, setOpenFaq] = useState<number | null>(0)

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
              Support Center
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Find answers, get help, and connect with our support team.
            </p>

            {/* Search */}
            <div className="max-w-2xl mx-auto relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for help..."
                className="w-full pl-12 pr-4 py-4 bg-surface border border-slate-700 rounded-xl text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
              />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Support Resources */}
      <section className="py-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {supportResources.map((resource, index) => (
              <motion.div
                key={resource.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Link href={resource.link}>
                  <Card variant="bordered" hover className="p-6 h-full text-center group">
                    <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${resource.color} flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform`}>
                      <resource.icon className="w-7 h-7 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-text-primary mb-2">
                      {resource.title}
                    </h3>
                    <p className="text-text-secondary text-sm">
                      {resource.description}
                    </p>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Submit a Ticket */}
      <section className="py-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold text-text-primary mb-6">
                Submit a Support Ticket
              </h2>
              <p className="text-text-secondary mb-8">
                Can&apos;t find what you&apos;re looking for? Submit a ticket and our support team will get back to you within 24 hours.
              </p>

              <form className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-text-primary mb-2">
                      Your Name
                    </label>
                    <input
                      type="text"
                      className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                      placeholder="John Doe"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-text-primary mb-2">
                      Email Address
                    </label>
                    <input
                      type="email"
                      className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                      placeholder="you@company.com"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-text-primary mb-2">
                    Category
                  </label>
                  <select className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200">
                    {ticketCategories.map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-text-primary mb-2">
                    Subject
                  </label>
                  <input
                    type="text"
                    className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                    placeholder="Brief description of your issue"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-text-primary mb-2">
                    Message
                  </label>
                  <textarea
                    className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200 resize-none"
                    rows={5}
                    placeholder="Describe your issue in detail..."
                  />
                </div>

                <Button size="lg" className="w-full">
                  Submit Ticket
                </Button>
              </form>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h2 className="text-3xl font-bold text-text-primary mb-6">
                Frequently Asked Questions
              </h2>
              <p className="text-text-secondary mb-8">
                Quick answers to common questions.
              </p>

              <div className="space-y-4">
                {faqs.map((faq, index) => (
                  <Card
                    key={index}
                    variant="bordered"
                    className="p-0 overflow-hidden"
                  >
                    <button
                      onClick={() => setOpenFaq(openFaq === index ? null : index)}
                      className="w-full p-4 flex items-center justify-between text-left"
                    >
                      <span className="font-medium text-text-primary">
                        {faq.question}
                      </span>
                      <Zap className={`w-5 h-5 text-text-secondary transition-transform duration-200 ${openFaq === index ? 'rotate-180' : ''}`} />
                    </button>
                    {openFaq === index && (
                      <div className="px-4 pb-4 text-text-secondary">
                        {faq.answer}
                      </div>
                    )}
                  </Card>
                ))}
              </div>

              <div className="mt-8 p-6 bg-surface/50 rounded-xl border border-slate-700">
                <h3 className="font-semibold text-text-primary mb-2">
                  Need urgent help?
                </h3>
                <p className="text-text-secondary text-sm mb-4">
                  For critical issues affecting production systems, call our emergency support line.
                </p>
                <a
                  href="tel:+15551234567"
                  className="text-primary font-medium flex items-center gap-2"
                >
                  <Phone className="w-4 h-4" />
                  +1 (555) 123-4567
                </a>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Status Page Link */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Check System Status
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                View real-time status of all KERNELIZE services and subscribe to incident notifications.
              </p>
              <a
                href="https://status.kernelize.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button size="lg">
                  View Status Page
                  <ExternalLink className="w-4 h-4 ml-2" />
                </Button>
              </a>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
