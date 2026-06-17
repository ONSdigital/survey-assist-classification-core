# Security Notice

## What?

This notice explains how vulnerabilities should be reported. At ONS there are
cyber security and information assurance teams, as well as security-conscious
people within the programmes, that assess and triage all reported
vulnerabilities.

## Scope

This disclosure policy applies only to vulnerabilities in the ONS products and services under the following conditions:

- ‘In scope’ vulnerabilities must be original, previously unreported, and not already discovered by internal procedures.
- Volumetric vulnerabilities are not in scope - meaning that simply overwhelming a service with a high volume of
  requests is not in scope.
- Reports of non-exploitable vulnerabilities, or reports indicating that our services do not fully align with 'best
  practice', for example missing security headers, are not in scope.
- TLS configuration weaknesses, for example 'weak'cipher suite support or the presence of TLS1.0 support, are not in
  scope.
- The policy applies to everyone, including for example the ONSstaff, third party suppliers and general users of the ONS
  publicservices.

## Reporting a Vulnerability

ONS advocates responsible vulnerability disclosure. If you've found a
vulnerability, we would like to know so we can fix it. Please report
vulnerabilities through the following channels:

- Contact the repository [code owners](CODEOWNERS)
- Submit a report through [hackerone](https://hackerone.com/52fa7bc0-5356-4c86-9f79-eeb03e1d55cc/embedded_submissions/new)

When reporting a vulnerability to us, please include:

- where the vulnerability can be observed
- a brief description of the vulnerability
- details of the steps we need to take to reproduce the vulnerability
- non-destructive exploitation details

If you are able to, please also include:

- the type of vulnerability, for example, the [OWASP](https://owasp.org/about/)
category
- screenshots or logs showing the exploitation of the vulnerability

If you are not sure if the vulnerability is genuine and exploitable, or you
have found:

- a non-exploitable vulnerability
- something you think could be improved

Then you can still reach out via email.

## Container Vulnerability Scanning

Once containers have been built they will be scanned for vulnerabilities and information will be updated here.

## Bug bounty

Unfortunately, ONS doesn't offer a paid bug bounty programme.

## Further Information

- https://www.gov.uk/help/report-vulnerability
- https://mojdigital.blog.gov.uk/vulnerability-disclosure-policy/
- https://www.ncsc.gov.uk/information/vulnerability-reporting
- https://github.com/Trewaters/security-README
