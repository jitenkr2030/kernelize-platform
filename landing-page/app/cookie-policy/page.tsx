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
import { Cookie, Shield, Settings } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'
import { useState } from 'react'

const cookieTypes = [
  {
    category: 'Essential Cookies',
    description: 'Required for the website and services to function properly. These cannot be disabled.',
    cookies: [
      { name: 'kernelize_session', purpose: 'Maintains your logged-in state', duration: 'Session' },
      { name: 'csrf_token', purpose: 'Protects against cross-site request forgery', duration: 'Session' },
      { name: 'auth_token', purpose: 'Stores authentication credentials', duration: '30 days' },
    ],
    required: true,
  },
  {
    category: 'Performance Cookies',
    description: 'Help us understand how visitors interact with our website by collecting anonymous information.',
    cookies: [
      { name: '_ga', purpose: 'Distinguishes unique users', duration: '2 years' },
      { name: '_gid', purpose: 'Groups user behavior', duration: '24 hours' },
      { name: '_gat', purpose: 'Throttles request rate', duration: '1 minute' },
    ],
    required: false,
  },
  {
    category: 'Functional Cookies',
    description: 'Enable enhanced functionality and personalization features.',
    cookies: [
      { name: 'language', purpose: 'Remembers your preferred language', duration: '1 year' },
      { name: 'theme', purpose: 'Stores your theme preference', duration: '1 year' },
      { name: 'last_project', purpose: 'Remembers your last accessed project', duration: '30 days' },
    ],
    required: false,
  },
  {
    category: 'Marketing Cookies',
    description: 'Used to track visitors across websites to display relevant advertisements.',
    cookies: [
      { name: 'ad_id', purpose: 'Tracks advertising campaigns', duration: '6 months' },
      { name: 'utm_source', purpose: 'Tracks marketing traffic sources', duration: '3 months' },
      { name: 'utm_medium', purpose: 'Tracks marketing medium', duration: '3 months' },
    ],
    required: false,
  },
]

export default function CookiePolicyPage() {
  const [showSettings, setShowSettings] = useState(false)

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
            <Cookie className="w-16 h-16 text-primary mx-auto mb-6" />
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-text-primary mb-6">
              Cookie Policy
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Last Updated: January 13, 2026
            </p>
            <Button onClick={() => setShowSettings(true)}>
              <Settings className="w-4 h-4 mr-2" />
              Manage Cookie Preferences
            </Button>
          </motion.div>
        </div>
      </section>

      {/* Introduction */}
      <section className="py-8">
        <div className="container-custom">
          <Card variant="bordered" className="p-8 max-w-3xl mx-auto">
            <h2 className="text-xl font-semibold text-text-primary mb-4">
              What Are Cookies?
            </h2>
            <p className="text-text-secondary leading-relaxed mb-4">
              Cookies are small text files that are stored on your device when you visit our website. They help us provide you with a better experience by enabling certain functionality, remembering your preferences, and analyzing how you use our services.
            </p>
            <p className="text-text-secondary leading-relaxed">
              This policy explains the types of cookies we use, why we use them, and your rights regarding their use. By continuing to use our services, you consent to the use of cookies as described in this policy.
            </p>
          </Card>
        </div>
      </section>

      {/* Cookie Categories */}
      <section className="py-8">
        <div className="container-custom">
          <div className="max-w-4xl mx-auto space-y-6">
            {cookieTypes.map((category, index) => (
              <motion.div
                key={category.category}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
                        {category.category}
                        {category.required && (
                          <span className="px-2 py-0.5 text-xs font-medium bg-primary/10 text-primary rounded">
                            Required
                          </span>
                        )}
                      </h3>
                      <p className="text-text-secondary text-sm mt-1">
                        {category.description}
                      </p>
                    </div>
                    {!category.required && (
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id={`cookie-${index}`}
                          defaultChecked={index < 2}
                          className="w-4 h-4 rounded border-slate-600 bg-surface text-primary focus:ring-primary focus:ring-offset-background"
                        />
                      </div>
                    )}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-2 px-3 text-text-secondary font-medium">Cookie Name</th>
                          <th className="text-left py-2 px-3 text-text-secondary font-medium">Purpose</th>
                          <th className="text-left py-2 px-3 text-text-secondary font-medium">Duration</th>
                        </tr>
                      </thead>
                      <tbody>
                        {category.cookies.map((cookie) => (
                          <tr key={cookie.name} className="border-b border-slate-700 last:border-0">
                            <td className="py-2 px-3 font-mono text-primary">{cookie.name}</td>
                            <td className="py-2 px-3 text-text-secondary">{cookie.purpose}</td>
                            <td className="py-2 px-3 text-text-secondary">{cookie.duration}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Managing Cookies */}
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
              Managing Your Cookie Preferences
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <Card variant="bordered" className="p-6 h-full text-center">
                <Settings className="w-12 h-12 text-primary mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  Browser Settings
                </h3>
                <p className="text-text-secondary text-sm">
                  Most browsers allow you to control cookies through their settings. You can usually find these options under "Privacy" or "Cookies" in your browser settings.
                </p>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card variant="bordered" className="p-6 h-full text-center">
                <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  Our Cookie Banner
                </h3>
                <p className="text-text-secondary text-sm">
                  When you first visit our website, you can customize your cookie preferences using our consent banner. Your choices are stored and respected.
                </p>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card variant="bordered" className="p-6 h-full text-center">
                <Cookie className="w-12 h-12 text-primary mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  Third-Party Tools
                </h3>
                <p className="text-text-secondary text-sm">
                  You can opt out of tracking by third-party advertising networks through tools like the Digital Advertising Alliance's opt-out page.
                </p>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Do Not Track */}
      <section className="py-8">
        <div className="container-custom">
          <Card variant="default" className="p-8 max-w-3xl mx-auto">
            <h2 className="text-xl font-semibold text-text-primary mb-4">
              Do Not Track
            </h2>
            <p className="text-text-secondary leading-relaxed">
              Some browsers offer a "Do Not Track" (DNT) signal. Currently, there is no industry standard for handling DNT signals, so we do not respond to DNT signals at this time. You can learn more about DNT at{' '}
              <a
                href="https://allaboutdnt.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:text-primary-hover"
              >
                allaboutdnt.com
              </a>.
            </p>
          </Card>
        </div>
      </section>

      {/* Updates */}
      <section className="py-8">
        <div className="container-custom">
          <Card variant="bordered" className="p-8 max-w-3xl mx-auto">
            <h2 className="text-xl font-semibold text-text-primary mb-4">
              Changes to This Policy
            </h2>
            <p className="text-text-secondary leading-relaxed">
              We may update this Cookie Policy from time to time to reflect changes in technology, legislation, or our business practices. We will notify you of any material changes by:

              • Posting the updated policy on this page
              • Updating the "Last Updated" date
              • Displaying a notice on our website

              We encourage you to review this policy periodically to stay informed about our use of cookies.
            </p>
          </Card>
        </div>
      </section>

      {/* Contact */}
      <section className="py-8 pb-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Questions About Cookies?
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                If you have any questions about our use of cookies, please contact our privacy team.
              </p>
              <Link href="/contact">
                <Button size="lg">
                  Contact Us
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </section>

      {/* Cookie Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-surface border border-slate-700 rounded-2xl max-w-lg w-full p-6 max-h-[80vh] overflow-y-auto"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-text-primary">
                Cookie Preferences
              </h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-text-secondary hover:text-text-primary transition-colors"
              >
                ✕
              </button>
            </div>
            <p className="text-text-secondary text-sm mb-6">
              Customize your cookie preferences below. Essential cookies cannot be disabled.
            </p>
            <div className="space-y-4 mb-6">
              {cookieTypes.slice(1).map((category, index) => (
                <div key={category.category} className="flex items-center justify-between">
                  <div>
                    <span className="font-medium text-text-primary">{category.category}</span>
                    <p className="text-xs text-text-secondary">{category.description}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" defaultChecked className="sr-only peer" />
                    <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-primary rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                  </label>
                </div>
              ))}
            </div>
            <div className="flex gap-4">
              <Button onClick={() => setShowSettings(false)} variant="secondary" className="flex-1">
                Save Preferences
              </Button>
              <Button onClick={() => setShowSettings(false)} className="flex-1">
                Accept All
              </Button>
            </div>
          </motion.div>
        </div>
      )}
    </main>
  )
}
