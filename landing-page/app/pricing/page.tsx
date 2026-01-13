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
import { Check, HelpCircle, ArrowUpRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const pricingPlans = [
  {
    name: 'Starter',
    price: 299,
    description: 'Perfect for small projects and individual developers',
    gradient: 'from-slate-500 to-slate-600',
    features: [
      'Up to 100GB bandwidth',
      '10 custom domains',
      'Basic analytics',
      'Email support',
      'Community access',
      '5 team members',
      'Standard API rate limits',
    ],
    cta: 'Start Free Trial',
    popular: false,
  },
  {
    name: 'Professional',
    price: 499,
    description: 'Ideal for growing teams and production applications',
    gradient: 'from-blue-500 to-indigo-600',
    features: [
      'Up to 1TB bandwidth',
      'Unlimited custom domains',
      'Advanced analytics & reports',
      'Priority email & chat support',
      'Team collaboration tools',
      'Custom integrations',
      '25 team members',
      'Higher API rate limits',
      'Custom domains with SSL',
      'Advanced security features',
    ],
    cta: 'Get Started',
    popular: true,
  },
  {
    name: 'Enterprise',
    price: null,
    description: 'For large organizations with advanced requirements',
    gradient: 'from-purple-500 to-violet-600',
    features: [
      'Unlimited bandwidth',
      'Unlimited custom domains',
      'Custom analytics & dashboards',
      '24/7 phone & dedicated support',
      'SLA guarantee',
      'Custom contracts',
      'On-premise deployment option',
      'Training & onboarding',
      'Unlimited team members',
      'Dedicated account manager',
      'Custom SLA & compliance',
      'White-glove onboarding',
    ],
    cta: 'Contact Sales',
    popular: false,
  },
]

const faqs = [
  {
    question: 'Can I change plans later?',
    answer: 'Yes, you can upgrade or downgrade your plan at any time. Changes will be reflected in your next billing cycle.',
  },
  {
    question: 'What payment methods do you accept?',
    answer: 'We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and wire transfers for annual Enterprise plans.',
  },
  {
    question: 'Is there a free trial?',
    answer: 'Yes! All paid plans come with a 30-day free trial. No credit card required to start.',
  },
  {
    question: 'What happens after my trial ends?',
    answer: 'You\'ll be prompted to choose a plan. If you don\'t select one, your account will be paused but your data will be preserved for 30 days.',
  },
  {
    question: 'Do you offer discounts for nonprofits?',
    answer: 'Yes, we offer 50% off for qualified nonprofit organizations. Contact our sales team to learn more.',
  },
  {
    question: 'What\'s included in the Enterprise plan?',
    answer: 'Enterprise plans include dedicated infrastructure, custom SLAs, compliance certifications (SOC 2, HIPAA), and white-glove onboarding. Contact sales for a custom quote.',
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
                className="relative w-14 h-8 bg-slate-700 rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
              >
                <motion.div
                  className="absolute top-1 w-6 h-6 bg-white rounded-full shadow-lg"
                  animate={{ left: billingCycle === 'monthly' ? '0.5rem' : 'calc(100% - 1.5rem)' }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              </button>
              <span className={`text-sm ${billingCycle === 'annual' ? 'text-text-primary' : 'text-text-secondary'}`}>
                Annual
                <span className="ml-2 text-xs text-green-500 font-medium">
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
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {pricingPlans.map((plan, index) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className={plan.popular ? 'md:-mt-4 md:mb-4' : ''}
              >
                <Card
                  variant={plan.popular ? 'default' : 'bordered'}
                  hover={!plan.popular}
                  className={`h-full p-6 ${plan.popular ? 'ring-2 ring-primary' : ''}`}
                >
                  {plan.popular && (
                    <div className="text-xs font-medium text-primary bg-primary/10 px-3 py-1 rounded-full inline-block mb-4">
                      Most Popular
                    </div>
                  )}
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${plan.gradient} flex items-center justify-center mb-4`}>
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
                        <span className="text-text-secondary ml-1">/month</span>
                      </div>
                    ) : (
                      <div className="text-2xl font-bold text-text-primary">
                        Custom Pricing
                      </div>
                    )}
                    {billingCycle === 'annual' && plan.price !== null && (
                      <p className="text-sm text-text-secondary mt-1">
                        ${Math.round(plan.price * 0.8 * 12)} billed annually
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
                      className="w-full"
                    >
                      {plan.cta}
                      {plan.price !== null && (
                        <ArrowUpRight className="w-4 h-4 ml-2" />
                      )}
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
      <section className="py-16">
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
