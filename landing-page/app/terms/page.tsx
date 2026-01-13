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
import { Zap, FileText } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import Link from 'next/link'

export default function TermsOfServicePage() {
  const sections = [
    {
      title: '1. Acceptance of Terms',
      content: `By accessing or using KERNELIZE's services, you agree to be bound by these Terms of Service and all applicable laws and regulations. If you do not agree with any part of these terms, you may not use our services.

Our services are intended for users who are at least 18 years of age. By using our services, you represent and warrant that you have the legal capacity to enter into this agreement.`,
    },
    {
      title: '2. Description of Services',
      content: `KERNELIZE provides a cloud infrastructure platform that allows users to deploy, manage, and scale applications and services. Our services include:

• Infrastructure provisioning and management
• Application deployment and scaling tools
• API and SDK access
• Monitoring and analytics
• Storage and database services
• Networking and security features

We reserve the right to modify, suspend, or discontinue any aspect of our services at any time with reasonable notice.`,
    },
    {
      title: '3. Account Registration',
      content: `To use our services, you must create an account and provide accurate, complete, and current information. You are responsible for:

• Maintaining the confidentiality of your account credentials
• All activities that occur under your account
• Notifying us immediately of any unauthorized use
• Ensuring all information remains accurate and up-to-date

We reserve the right to suspend or terminate accounts that violate these terms or engage in fraudulent or illegal activities.`,
    },
    {
      title: '4. Acceptable Use',
      content: `You agree not to use our services to:

• Violate any laws or regulations
• Infringe on the rights of others
• Transmit malware, viruses, or harmful code
• Engage in unauthorized access to systems or networks
• Interfere with or disrupt our services
• Use our services for illegal or harmful purposes
• Resell or redistribute our services without authorization
• Engage in activities that could damage, disable, or impair our infrastructure

Violations may result in immediate termination of your account and access to our services.`,
    },
    {
      title: '5. Subscription and Billing',
      content: `Some aspects of our services require payment. By subscribing to a paid plan, you agree to:

• Pay all fees associated with your selected plan
• Provide valid payment information
• Allow us to charge your payment method on a recurring basis
• Notify us of any changes to your billing information

Prices are subject to change with 30 days' notice. Failure to pay may result in suspension or termination of your account.

All fees are non-refundable unless otherwise specified or required by law.`,
    },
    {
      title: '6. Intellectual Property',
      content: `You retain ownership of all content and data you upload to our services. By uploading content, you grant us a limited license to use, process, and display your content solely to provide our services.

KERNELIZE's intellectual property includes:

• Our platform, software, and technology
• Brand names, logos, and trademarks
• Documentation and educational materials
• User interface designs and graphics

You may not use our trademarks or intellectual property without our written consent.`,
    },
    {
      title: '7. User Content',
      content: `You retain full ownership of all content, data, and materials you upload, process, or store using our services ("User Content"). We do not claim ownership of your User Content.

By using our services, you grant us a worldwide, royalty-free license to:

• Use your User Content solely to provide our services
• Make backup copies for disaster recovery
• Process your User Content as necessary to provide functionality
• Comply with legal requirements

You are solely responsible for ensuring you have all necessary rights to the User Content you upload.`,
    },
    {
      title: '8. Service Level Agreement',
      content: `For paid subscription plans, we guarantee 99.99% monthly uptime for our platform availability, measured and calculated as described in our SLA documentation.

In the event of a service outage that falls below our SLA guarantee, eligible customers may receive service credits as outlined in our SLA terms. Service credits are your sole and exclusive remedy for service outages.

The SLA does not apply to:

• Scheduled maintenance (with advance notice)
• Circumstances beyond our reasonable control
• Issues caused by your configuration or third-party services`,
    },
    {
      title: '9. Disclaimers',
      content: `OUR SERVICES ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED. WE DO NOT WARRANT THAT:

• Our services will be uninterrupted, error-free, or secure
• Any errors or defects will be corrected
• Our services will meet your specific requirements
• The results obtained from our services will be accurate

We do not warrant the accuracy or reliability of any content, information, or materials obtained through our services.`,
    },
    {
      title: '10. Limitation of Liability',
      content: `TO THE MAXIMUM EXTENT PERMITTED BY LAW, KERNELIZE AND ITS AFFILIATES SHALL NOT BE LIABLE FOR:

• Any indirect, incidental, special, consequential, or punitive damages
• Loss of profits, revenue, data, or business opportunities
• Service interruptions or data loss
• Actions of third parties

Our total liability shall not exceed the greater of (a) the amounts paid by you in the 12 months preceding the claim, or (b) $1,000.

These limitations apply even if we have been advised of the possibility of such damages.`,
    },
    {
      title: '11. Indemnification',
      content: `You agree to defend, indemnify, and hold harmless KERNELIZE and its affiliates, officers, directors, employees, and agents from and against any and all claims, damages, losses, costs, and expenses (including reasonable attorneys' fees) arising from:

• Your use of our services
• Your violation of these Terms of Service
• Your violation of any rights of a third party
• Your User Content

We will provide prompt written notice of any such claim and cooperate with your defense.`,
    },
    {
      title: '12. Termination',
      content: `Either party may terminate this agreement at any time:

• You may cancel your account at any time through your account settings
• We may suspend or terminate your account for violations of these terms

Upon termination:

• You will remain responsible for all fees incurred
• We will provide reasonable time to export your data (typically 30 days)
• We may delete your data after the export period
• Certain provisions will survive termination (intellectual property, confidentiality, limitation of liability)`,
    },
    {
      title: '13. Governing Law',
      content: `These Terms of Service shall be governed by and construed in accordance with the laws of the State of California, United States, without regard to its conflict of law provisions.

Any disputes arising from these terms or your use of our services shall be resolved through binding arbitration in San Francisco, California, under the rules of the American Arbitration Association.

If arbitration is not permitted or if you are a consumer in a jurisdiction that requires a different dispute resolution process, you may bring claims in the courts of San Francisco, California.`,
    },
    {
      title: '14. Changes to Terms',
      content: `We may modify these Terms of Service at any time. Material changes will be communicated via:

• Email notification to registered users
• Notice in our dashboard
• Posting on our website at least 30 days before effective date

Your continued use of our services after any modifications indicates your acceptance of the updated terms. If you do not agree to the changes, you must discontinue use of our services.`,
    },
    {
      title: '15. General Provisions',
      content: `• Entire Agreement: These Terms of Service constitute the entire agreement between you and KERNELIZE regarding our services.

• Severability: If any provision is found unenforceable, the remaining provisions remain in effect.

• Waiver: Failure to enforce any right under these terms does not constitute a waiver of that right.

• Assignment: You may not assign these terms without our consent. We may assign our rights and obligations freely.

• Notices: All notices should be sent to legal@kernelize.com.`,
    },
    {
      title: '16. Contact Information',
      content: `If you have questions about these Terms of Service, please contact us at:

KERNELIZE, Inc.
100 Technology Drive
San Francisco, CA 94105
Email: legal@kernelize.com
Phone: +1 (555) 123-4567`,
    },
  ]

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
            <FileText className="w-16 h-16 text-primary mx-auto mb-6" />
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-text-primary mb-6">
              Terms of Service
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto">
              Last Updated: January 13, 2026
            </p>
          </motion.div>
        </div>
      </section>

      {/* Table of Contents */}
      <section className="py-8">
        <div className="container-custom">
          <Card variant="bordered" className="p-6 max-w-3xl mx-auto">
            <h2 className="text-lg font-semibold text-text-primary mb-4">
              Table of Contents
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {sections.map((section, index) => {
                const id = section.title.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')
                return (
                  <a
                    key={index}
                    href={`#${id}`}
                    className="text-sm text-text-secondary hover:text-primary transition-colors flex items-center gap-2"
                  >
                    <span className="text-primary">{index + 1}.</span>
                    {section.title.replace(/^\d+\.\s*/, '')}
                  </a>
                )
              })}
            </div>
          </Card>
        </div>
      </section>

      {/* Content */}
      <section className="py-8 pb-24">
        <div className="container-custom">
          <div className="max-w-3xl mx-auto space-y-8">
            {sections.map((section, index) => {
              const id = section.title.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')
              return (
                <motion.div
                  key={index}
                  id={id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5 }}
                >
                  <h2 className="text-2xl font-bold text-text-primary mb-4">
                    {section.title}
                  </h2>
                  <div className="text-text-secondary leading-relaxed whitespace-pre-line">
                    {section.content}
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Related Pages */}
      <section className="py-8 pb-16">
        <div className="container-custom">
          <div className="max-w-3xl mx-auto text-center">
            <p className="text-text-secondary mb-4">
              Related legal documents:
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link href="/privacy" className="text-primary hover:text-primary-hover transition-colors">
                Privacy Policy
              </Link>
              <span className="text-text-secondary">•</span>
              <Link href="/cookie-policy" className="text-primary hover:text-primary-hover transition-colors">
                Cookie Policy
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}
