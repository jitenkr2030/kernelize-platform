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
import { Check, HelpCircle } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const pricingPlans = [
  {
    name: 'Starter',
    price: 299,
    description: 'Perfect for small to medium businesses and startups getting started with data integration',
    gradient: 'from-emerald-500 to-teal-600',
    features: [
      'Up to 1TB/month data processing',
      '10 third-party connectors',
      '25 automated workflows',
      '100GB compressed data storage',
      'Email support & documentation',
      'Up to 5 team members',
      'Basic analytics dashboard',
    ],
    cta: 'Start Free Trial',
    popular: false,
  },
  {
    name: 'Professional',
    price: 999,
    description: 'Ideal for growing enterprises and data teams requiring advanced capabilities',
    gradient: 'from-primary to-purple-500',
    features: [
      'Up to 10TB/month data processing',
      '25+ connectors (Snowflake, Databricks, Tableau, Power BI)',
      'Unlimited workflows with custom triggers',
      '1TB compressed data storage',
      'SSO, RBAC, audit logging',
      'Advanced compression analytics & cost optimization',
      'Priority email & chat support',
      'Up to 25 team members',
    ],
    cta: 'Start Free Trial',
    popular: true,
  },
  {
    name: 'Enterprise',
    price: 2999,
    description: 'For large enterprises and regulated industries requiring maximum performance and compliance',
    gradient: 'from-amber-500 to-orange-500',
    features: [
      'Unlimited data processing (fair usage policy)',
      'Full marketplace access + custom connectors',
      'Advanced workflow automation with AI',
      'Unlimited compressed data storage',
      'Full compliance (GDPR, HIPAA, SOC 2), encryption at rest',
      'Multi-region deployment, high availability',
      'Custom BI dashboards, predictive analytics',
      '24/7 phone support, dedicated account manager',
      'Unlimited team members',
    ],
    cta: 'Contact Sales',
    popular: false,
  },
  {
    name: 'Enterprise Plus',
    price: null,
    description: 'Fortune 500 and government organizations requiring custom deployment and dedicated services',
    gradient: 'from-red-500 to-rose-600',
    features: [
      'Everything in Enterprise Plan',
      'Custom connector development',
      'On-premise deployment (private cloud)',
      'Professional services (implementation, training, consulting)',
      '99.99% uptime SLA guarantee',
      'Annual commitments with volume discounts',
      'Custom contracts & dedicated resources',
    ],
    cta: 'Contact Sales',
    popular: false,
  },
]

const faqs = [
  {
    question: 'What payment methods do you accept?',
    answer: 'We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and wire transfers for Enterprise plans. Annual contracts can be paid via invoice with net-30 terms.',
  },
  {
    question: 'Is there a free trial available?',
    answer: 'Yes! All paid plans come with a comprehensive 30-day free trial with full access to all features. No credit card required to start. Experience the full power of KERNELIZE before committing.',
  },
  {
    question: 'What happens when I exceed my plan limits?',
    answer: 'We provide overage protection with flexible scaling. For usage beyond plan limits, you can upgrade your plan at any time or utilize our usage-based pricing model: $0.05/GB for data compression, $0.02/GB for transformations, and $0.10/GB-month for additional storage.',
  },
  {
    question: 'Can I switch plans as my needs evolve?',
    answer: 'Absolutely. You can upgrade or downgrade your plan at any time. Upgrades take effect immediately with prorated billing. Downgrades apply at the end of your current billing cycle. Our team is here to help you find the optimal plan for your needs.',
  },
  {
    question: 'Do you offer on-premise deployment?',
    answer: 'Yes, our Enterprise Plus plan includes on-premise deployment options for organizations that require data to remain on their own infrastructure. Contact our sales team for more information.',
  },
  {
    question: 'What support options are included?',
    answer: 'Starter plans include email support and documentation. Professional plans add priority email and chat support. Enterprise plans include 24/7 phone support with a dedicated account manager. Enterprise Plus includes all of the above plus dedicated professional services.',
  },
]

export default function PricingPage() {
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('monthly')
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
              Simple, Transparent
              <br />
              <span className="gradient-text">Pricing</span>
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Choose the perfect plan for your needs. No hidden fees, no surprises.
            </p>

            {/* Billing Toggle */}
            <div className="flex items-center justify-center gap-4 mb-8">
              <span className={`text-sm ${billingCycle === 'monthly' ? 'text-text-primary' : 'text-text-secondary'}`}>
                Monthly
              </span>
              <button
                onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'annual' : 'monthly')}
                className="relative w-14 h-7 bg-white/10 border border-white/20 rounded-full transition-colors hover:border-primary/50"
              >
                <motion.div
                  className="absolute top-1 w-5 h-5 rounded-full bg-gradient-to-r from-primary to-purple-500"
                  animate={{ x: billingCycle === 'monthly' ? 4 : 25 }}
                  transition={{ duration: 0.2 }}
                />
              </button>
              <span className={`text-sm ${billingCycle === 'annual' ? 'text-text-primary' : 'text-text-secondary'}`}>
                Annual
                <span className="ml-2 px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full">
                  Save 20%
                </span>
              </span>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Pricing Cards */}
      <section className="py-8 pb-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 max-w-7xl mx-auto">
            {pricingPlans.map((plan, index) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className={plan.popular ? 'xl:-mt-6' : ''}
              >
                <Card
                  variant={plan.popular ? 'default' : 'bordered'}
                  className={`h-full p-6 relative overflow-hidden ${
                    plan.popular ? 'border-2 border-primary shadow-xl shadow-primary/20' : ''
                  }`}
                >
                  {/* Popular glow */}
                  {plan.popular && (
                    <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5" />
                  )}
                  {/* Top gradient bar */}
                  <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${plan.gradient}`} />
                  
                  <div className="relative z-10">
                    {plan.popular && (
                      <div className="text-xs font-medium text-primary bg-primary/10 px-3 py-1 rounded-full inline-block mb-4">
                        Most Popular
                      </div>
                    )}
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${plan.gradient} flex items-center justify-center mb-4 ${plan.popular ? 'shadow-lg' : ''}`}>
                      <span className="text-white font-bold text-lg">
                        {plan.name.charAt(0)}
                      </span>
                    </div>
                    <h3 className="text-xl font-semibold text-text-primary mb-2">
                      {plan.name}
                    </h3>
                    <p className="text-text-secondary text-sm mb-6">
                      {plan.description}
                    </p>

                    {/* Price */}
                    <div className="mb-6">
                      {plan.price !== null ? (
                        <div className="flex items-baseline">
                          <span className="text-4xl font-bold text-text-primary">
                            ${billingCycle === 'annual' ? Math.round(plan.price * 0.8) : plan.price}
                          </span>
                          <span className="text-text-secondary ml-2">/month</span>
                        </div>
                      ) : (
                        <div className="text-2xl font-bold text-text-primary">
                          Custom
                        </div>
                      )}
                      {billingCycle === 'annual' && plan.price !== null && (
                        <p className="text-sm text-text-secondary mt-1">
                          Billed ${Math.round(plan.price * 0.8 * 12)}/year
                        </p>
                      )}
                    </div>

                    {/* CTA Button */}
                    <Link
                      href={plan.price !== null ? '/signup' : '#contact'}
                      className="block mb-6"
                    >
                      <Button
                        variant={plan.popular ? 'primary' : 'secondary'}
                        className="w-full py-3"
                      >
                        {plan.cta}
                      </Button>
                    </Link>

                    {/* Features */}
                    <ul className="space-y-3">
                      {plan.features.map((feature) => (
                        <li key={feature} className="flex items-start gap-3">
                          <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                          <span className="text-text-secondary text-sm">
                            {feature}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Money Back Guarantee */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="bordered" className="p-8 md:p-12">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-2">
                  30-Day Money-Back Guarantee
                </h3>
                <p className="text-text-secondary">
                  Try KERNELIZE risk-free. If you&apos;re not completely satisfied, we&apos;ll refund your payment.
                </p>
              </div>
              <Link href="/signup">
                <Button size="lg">
                  Start Free Trial
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </section>

      {/* FAQ Section */}
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
              Frequently Asked Questions
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Have questions? We&apos;ve got answers. Can&apos;t find what you&apos;re looking for? Contact our support team.
            </p>
          </motion.div>

          <div className="max-w-3xl mx-auto space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card
                  variant="bordered"
                  className="p-0 overflow-hidden"
                >
                  <button
                    onClick={() => setOpenFaq(openFaq === index ? null : index)}
                    className="w-full p-6 flex items-center justify-between text-left"
                  >
                    <span className="font-medium text-text-primary">
                      {faq.question}
                    </span>
                    <HelpCircle className={`w-5 h-5 text-text-secondary transition-transform duration-200 ${openFaq === index ? 'rotate-180' : ''}`} />
                  </button>
                  {openFaq === index && (
                    <div className="px-6 pb-6 text-text-secondary">
                      {faq.answer}
                    </div>
                  )}
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact CTA */}
      <section className="py-16 pb-24">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
                Need a Custom Solution?
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                Get a tailored quote for your organization&apos;s specific needs. Our enterprise team is here to help.
              </p>
              <Link href="/contact">
                <Button size="lg">
                  Contact Sales
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
