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

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, Mail, MapPin, Phone, MessageSquare, Send, ArrowRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const contactInfo = [
  {
    icon: Mail,
    title: 'Email',
    value: 'contact@kernelize.com',
    href: 'mailto:contact@kernelize.com',
  },
  {
    icon: Phone,
    title: 'Phone',
    value: '+1 (555) 123-4567',
    href: 'tel:+15551234567',
  },
  {
    icon: MapPin,
    title: 'Office',
    value: '100 Technology Drive, San Francisco, CA 94105',
    href: 'https://maps.google.com/?q=100+Technology+Drive+San+Francisco+CA',
  },
]

const departments = [
  'Sales',
  'Support',
  'Partnerships',
  'Press/Media',
  'Careers',
  'Security',
  'Other',
]

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    department: 'Sales',
    message: '',
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1500))
    setIsSubmitting(false)
    setSubmitted(true)
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
              Get in Touch
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Have questions? We&apos;d love to hear from you. Our team is here to help.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Contact Form & Info */}
      <section className="py-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {/* Contact Form */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="lg:col-span-2"
            >
              <Card variant="bordered" className="p-8">
                {!submitted ? (
                  <>
                    <h2 className="text-2xl font-bold text-text-primary mb-6">
                      Send Us a Message
                    </h2>
                    <form onSubmit={handleSubmit} className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <label className="block text-sm font-medium text-text-primary mb-2">
                            Your Name *
                          </label>
                          <input
                            type="text"
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                            placeholder="John Doe"
                            required
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-text-primary mb-2">
                            Email Address *
                          </label>
                          <input
                            type="email"
                            value={formData.email}
                            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                            className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                            placeholder="you@company.com"
                            required
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <label className="block text-sm font-medium text-text-primary mb-2">
                            Company
                          </label>
                          <input
                            type="text"
                            value={formData.company}
                            onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                            className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                            placeholder="Your Company"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-text-primary mb-2">
                            Department *
                          </label>
                          <select
                            value={formData.department}
                            onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                            className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                          >
                            {departments.map((dept) => (
                              <option key={dept} value={dept}>
                                {dept}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-text-primary mb-2">
                          Message *
                        </label>
                        <textarea
                          value={formData.message}
                          onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                          className="w-full px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200 resize-none"
                          rows={5}
                          placeholder="How can we help you?"
                          required
                        />
                      </div>

                      <Button type="submit" size="lg" disabled={isSubmitting}>
                        {isSubmitting ? (
                          'Sending...'
                        ) : (
                          <>
                            Send Message
                            <Send className="w-4 h-4 ml-2" />
                          </>
                        )}
                      </Button>
                    </form>
                  </>
                ) : (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-4">
                      <MessageSquare className="w-8 h-8 text-green-500" />
                    </div>
                    <h2 className="text-2xl font-bold text-text-primary mb-2">
                      Message Sent!
                    </h2>
                    <p className="text-text-secondary mb-6">
                      Thank you for reaching out. We&apos;ll get back to you within 24 hours.
                    </p>
                    <Button onClick={() => setSubmitted(false)} variant="secondary">
                      Send Another Message
                    </Button>
                  </div>
                )}
              </Card>
            </motion.div>

            {/* Contact Info */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="space-y-6"
            >
              <h2 className="text-2xl font-bold text-text-primary">
                Contact Information
              </h2>
              <p className="text-text-secondary">
                Prefer to reach out directly? Here&apos;s how you can get in touch with us.
              </p>

              <div className="space-y-4">
                {contactInfo.map((info) => (
                  <Card key={info.title} variant="bordered" className="p-4">
                    <a
                      href={info.href}
                      target={info.href.startsWith('http') ? '_blank' : undefined}
                      rel={info.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                      className="flex items-center gap-4 group"
                    >
                      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <info.icon className="w-6 h-6 text-primary" />
                      </div>
                      <div>
                        <div className="font-medium text-text-primary">
                          {info.title}
                        </div>
                        <div className="text-text-secondary group-hover:text-primary transition-colors">
                          {info.value}
                        </div>
                      </div>
                    </a>
                  </Card>
                ))}
              </div>

              <Card variant="default" className="p-6">
                <h3 className="font-semibold text-text-primary mb-3">
                  Support Hours
                </h3>
                <div className="space-y-2 text-sm text-text-secondary">
                  <div className="flex justify-between">
                    <span>Monday - Friday</span>
                    <span>9:00 AM - 6:00 PM PST</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Saturday - Sunday</span>
                    <span>Closed</span>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Link */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Have Technical Questions?
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                Check out our documentation and support resources for quick answers to common questions.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/docs">
                  <Button size="lg">
                    View Documentation
                  </Button>
                </Link>
                <Link href="/support">
                  <Button variant="secondary" size="lg">
                    Visit Support Center
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
