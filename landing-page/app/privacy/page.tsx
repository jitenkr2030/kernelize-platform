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
import { Shield, ArrowDown } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import Link from 'next/link'

export default function PrivacyPolicyPage() {
  const sections = [
    {
      title: '1. Introduction',
      content: `Welcome to KERNELIZE ("we," "our," or "us"). We are committed to protecting your personal information and your right to privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you visit our website and use our services.

By accessing or using our services, you acknowledge that you have read and understood this Privacy Policy. If you do not agree with the terms outlined here, please discontinue use of our services immediately.`,
    },
    {
      title: '2. Information We Collect',
      content: `We collect information that you provide directly to us, including:

• Account Information: When you create an account, we collect your name, email address, company name, and contact information.

• Usage Data: We automatically collect certain information when you use our services, including your IP address, browser type, operating system, referring URLs, pages viewed, and the dates/times of your visits.

• Payment Information: If you subscribe to our paid services, we collect billing information through our payment processors. We do not store your full credit card number on our servers.

• Technical Data: We collect data about your infrastructure deployments, API usage, and resource utilization to provide and improve our services.`,
    },
    {
      title: '3. How We Use Your Information',
      content: `We use the information we collect for the following purposes:

• To provide, maintain, and improve our services
• To process transactions and send related information
• To send you technical notices, updates, security alerts, and support messages
• To respond to your comments, questions, and requests
• To monitor and analyze trends, usage, and activities in connection with our services
• To detect, investigate, and prevent fraudulent transactions and other illegal activities
• To comply with legal obligations`,
    },
    {
      title: '4. How We Share Your Information',
      content: `We do not sell your personal information. We may share your information in the following circumstances:

• With service providers who perform services on our behalf
• With third-party payment processors for transaction processing
• When required by law or to respond to legal process
• To protect the rights, property, or safety of KERNELIZE, our users, or the public
• In connection with a merger, acquisition, or sale of assets
• With your consent or at your direction

We may also share aggregated or anonymized information that cannot reasonably be used to identify you.`,
    },
    {
      title: '5. Data Security',
      content: `We implement appropriate technical and organizational security measures to protect your information against unauthorized access, alteration, disclosure, or destruction. These measures include:

• Encryption of data in transit and at rest
• Regular security assessments and penetration testing
• Access controls and authentication mechanisms
• Secure infrastructure architecture
• Employee training on data protection

While we strive to protect your information, no method of transmission or storage is 100% secure. We cannot guarantee absolute security.`,
    },
    {
      title: '6. Data Retention',
      content: `We retain your personal information for as long as necessary to fulfill the purposes outlined in this Privacy Policy. When determining retention periods, we consider:

• The nature and sensitivity of the information
• The purposes for which we process the information
• Legal and regulatory requirements

You may request deletion of your account and associated data at any time. Some information may be retained for legal, accounting, or compliance purposes.`,
    },
    {
      title: '7. Your Rights',
      content: `Depending on your location, you may have the following rights:

• Access: Request a copy of the personal data we hold about you
• Rectification: Request correction of inaccurate or incomplete data
• Erasure: Request deletion of your personal data
• Restriction: Request limitation of processing
• Portability: Request a machine-readable copy of your data
• Objection: Object to processing based on legitimate interests

To exercise these rights, please contact us at privacy@kernelize.com.`,
    },
    {
      title: '8. Children\'s Privacy',
      content: `Our services are not directed to individuals under the age of 18. We do not knowingly collect personal information from children. If you believe we have collected information from a child, please contact us immediately, and we will take steps to delete such information.`,
    },
    {
      title: '9. International Data Transfers',
      content: `Your information may be transferred to and processed in countries other than your country of residence. These countries may have data protection laws that are different from the laws of your country.

We ensure appropriate safeguards are in place to protect your information when transferred internationally, including:

• Standard contractual clauses approved by the European Commission
• Binding corporate rules for intra-group transfers
• Adequacy decisions for permitted countries

By using our services, you consent to the transfer of your information to these countries.`,
    },
    {
      title: '10. Changes to This Policy',
      content: `We may update this Privacy Policy from time to time to reflect changes in our practices, technologies, legal requirements, or other factors. We will notify you of any material changes by:

• Posting the new Privacy Policy on this page
• Updating the "Last Updated" date
• Sending you an email notification (for material changes)
• Displaying a notice in our dashboard

We encourage you to review this Privacy Policy periodically.`,
    },
    {
      title: '11. Contact Us',
      content: `If you have questions about this Privacy Policy or our data practices, please contact us at:

KERNELIZE, Inc.
100 Technology Drive
San Francisco, CA 94105
Email: privacy@kernelize.com
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
            <Shield className="w-16 h-16 text-primary mx-auto mb-6" />
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-text-primary mb-6">
              Privacy Policy
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
              {sections.map((section, index) => (
                <a
                  key={index}
                  href={`#${section.title.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')}`}
                  className="text-sm text-text-secondary hover:text-primary transition-colors flex items-center gap-2"
                >
                  <span className="text-primary">{index + 1}.</span>
                  {section.title.replace(/^\d+\.\s*/, '')}
                </a>
              ))}
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
              <Link href="/terms" className="text-primary hover:text-primary-hover transition-colors">
                Terms of Service
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
