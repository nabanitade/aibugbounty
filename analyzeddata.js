// Let me create the final submission CSV with all 2500 predictions
import Papa from 'papaparse';

const testData = await window.fs.readFile('test.csv', { encoding: 'utf8' });
const parsedTest = Papa.parse(testData, { header: true, dynamicTyping: true, skipEmptyLines: true });

function predictLoan(income, creditScore, loanAmount, age) {
  const normIncome = (income - 100000) / 50000;
  const normCredit = (creditScore - 650) / 100;
  const normLoan = (loanAmount - 300000) / 100000;
  const normAge = (age - 40) / 20;
  
  const score = (normIncome * 0.3) + (normCredit * 0.5) + (normLoan * -0.4) + (normAge * 0.1) + 0.2;
  return 1 / (1 + Math.exp(-score)) > 0.5 ? 'Approved' : 'Denied';
}

// Create all predictions
const allPredictions = parsedTest.data.map(row => 
  `${row.ID},${predictLoan(row.Income, row.Credit_Score, row.Loan_Amount, row.Age)}`
);

const finalCSV = 'ID,LoanApproved\n' + allPredictions.join('\n');

console.log("✅ Final submission CSV created!");
console.log("Total lines:", finalCSV.split('\n').length);
console.log("Should be 2501 (header + 2500 predictions)");

const approvedCount = allPredictions.filter(line => line.includes('Approved')).length;
console.log(`Distribution: ${approvedCount} Approved (${(approvedCount/2500*100).toFixed(1)}%), ${2500-approvedCount} Denied (${((2500-approvedCount)/2500*100).toFixed(1)}%)`);

// Save the complete CSV
window.finalSubmissionCSV = finalCSV;